import collections
import io
import os
import typing
import torch

import numpy as np

from .operators import CommonGraph, ExtendedOperator, GraphOptimizer, HybridQuantizer, HalfQuantizer
from .operators.op_version import OPVersioner
from .operators.tflite import Tensor
from .operators.torch import OPERATOR_CONVERTER_DICT
from .operators.torch.base import NoTrackOperator, TrackQParamsOperator
from .operators.torch.aten import ATenDequantizeOperator
from ..util.converter_util import generate_converter_config
from ..util.util import get_logger

log = get_logger(__name__, 'INFO')


class TFLiteConverter(object):
    def __init__(
        self,
        model: typing.Union[torch.jit.ScriptFunction, torch.jit.ScriptModule, torch.nn.Module],
        dummy_input: typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]],
        tflite_path: str,
        input_transpose: typing.Optional[typing.Union[bool, typing.Iterable[bool]]] = None,
        output_transpose: typing.Optional[typing.Union[bool, typing.Iterable[bool]]] = None,
        nchw_transpose: bool = True,
        dump_jit_model_path: typing.Optional[str] = None,
        dump_dummy_input_path: typing.Optional[str] = None,
        dump_config_path: typing.Optional[str] = None,
        strict_symmetric_check: bool = False,
        preserve_tensors: bool = False,
        optimize: int = GraphOptimizer.ALL_OPTIMIZE,
        quantize_target_type: str = 'uint8',
        quantize_input_output_type: typing.Optional[str] = None,
        hybrid_quantization_from_float: bool = False,
        hybrid_per_channel: bool = False,
        hybrid_asymmetric_inputs: bool = True,
        hybrid_quantize_weight_type: typing.Optional[str] = None,
        fuse_quant_dequant: bool = False,
        fuse_input_indices: typing.Optional[typing.List[int]] = None,
        fuse_output_indices: typing.Optional[typing.List[int]] = None,
        gc_when_reload: bool = False,
        group_conv_rewrite: bool = False,
        rewrite_quantizable: bool = False,
        tflite_micro_rewrite: bool = False,
        map_bilstm_to_lstm: bool = False,
        float16_quantization: bool = False,
        enable_mtk_ops: bool = False,
        max_transpose_dims: int = -1,
        hybrid_conv: bool = True,
    ) -> None:
        """ The TFLiteConverter class

        Args:
            model (typing.Union[torch.jit.ScriptFunction, torch.jit.ScriptModule, torch.nn.Module]): The input model \
                (either traced or non-traced)
            dummy_input (typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]]): A viable input to the model
            tflite_path (str): Path to use for exporting
            input_transpose (typing.Optional[typing.Union[bool, typing.Iterable[bool]]], optional): Whether to \
                transpose the input(s). Defaults to None(True for 4d-input, False otherwise).
            output_transpose (typing.Optional[typing.Union[bool, typing.Iterable[bool]]], optional): Whether to \
                transpose the output(s). Defaults to None(True for 4d-input, False otherwise).
            nchw_transpose (bool): Whether to perform nchw->nhwc transposes on input and output tensors. \
                `False` is specified, the arguments `input_transpose` and `output_transpose` will be ignored.
            dump_jit_model_path (typing.Optional[str]): The path for dumping the jit model. Defaults to None
            dump_dummy_input_path (typing.Optional[str]): The path for dumping the dummy input. Defaults to None
            dump_config_path (typing.Optional[str]): The path for dumping the json config. Defaults to None
            strict_symmetric_check (bool): Strict symmetric quantization checks. Defaults to False
            preserve_tensors (bool): Preserve the copies of the intermediate tensors. Defaults to False
            optimize (int): The level of graph optimization. Defaults to `GraphOptimizer.ALL_OPTIMIZE`
            quantize_target_type (str): Target type for quantization. Defaults to 'uint8'
            quantize_input_output_type (str): Input and output type for quantization. Defaults to None (inferred)
            hybrid_quantization_from_float (bool): Direct hybrid quantization from a float model. Defaults to False
            hybrid_per_channel (bool): Prefer per-channel kernels in hybrid quantization. Defaults to False
            hybrid_asymmetric_inputs (bool): Prefer asymmetric inputs while performing hybrid quantization
            hybrid_quantize_weight_type (typing.Optional[str]): Quantized weight type for hybrid quantization. \
                If it is unset, then the value of `quantize_target_type` will be used. Defaults to None
            fuse_quant_dequant (bool): Remove quant and dequant nodes directly connected to i/o nodes. Defaults to False
            fuse_input_indices (typing.Optional[typing.List[int]]): Used together with `fuse_quant_dequant`. Indices \
                of input nodes to fuse with `Quantize`. Defaults to None (which fuses all inputs available)
            fuse_output_indices (typing.Optional[typing.List[int]]): Used together with `fuse_quant_dequant`. Indices \
                of output nodes to fuse with `Dequantize`. Defaults to None (which fuses all outputs available)
            gc_when_reload (bool): Apply GC when reloading the torchscript into memory
            group_conv_rewrite (bool): Rewriting for group [de]convolution. Defaults to False
            rewrite_quantizable (bool): Rewriting quantizable ops (e.g. BATCH_MATMUL, SOFTMAX, LOG_SOFTMAX) \
                to use quantized kernels. Defaults to False
            tflite_micro_rewrite (bool): Rewriting for running on TFLite-micro. Defaults to False
            map_bilstm_to_lstm (bool): Translating bidirectional LSTM to TFLite ops with `UnidirectionalLSTM`. \
                Defaults to False
            float16_quantization (bool): Quantize constants with float32 dtype to floa16 dtype. Defaults to False
            enable_mtk_ops (bool): Translating with custom MTK operators. Defaults to False
            max_transpose_dims (int): Max dimensions for the `Transpose` op. Defaults to -1, which means unlimited
            hybrid_conv (bool): Enable hybrid quantization for Conv2d and DepthwiseConv2d. Defaults to True
        """

        self.model = model
        self.lower_model = None
        self.graph = None
        self.tensor_map = {}
        self.tensor_map_copies = {}
        self.common_graph = CommonGraph()

        if type(dummy_input) in (tuple, list):
            self.dummy_input = dummy_input
        else:
            self.dummy_input = [dummy_input]

        self.tflite_path = tflite_path
        self.nchw_transpose = nchw_transpose

        if self.nchw_transpose:
            self.input_transpose = input_transpose
            self.output_transpose = output_transpose
        else:
            self.input_transpose = False
            self.output_transpose = False

        self.strict_symmetric_check = strict_symmetric_check

        self.dump_jit_model_path = dump_jit_model_path
        self.dump_dummy_input_path = dump_dummy_input_path
        self.dump_config_path = dump_config_path
        self.preserve_tensors = preserve_tensors
        self.optimize = optimize
        self.hybrid = hybrid_quantization_from_float
        self.hybrid_per_channel = hybrid_per_channel
        self.hybrid_asymmetric_inputs = hybrid_asymmetric_inputs
        self.fuse_quant_dequant = fuse_quant_dequant
        self.fuse_input_indices = fuse_input_indices
        self.fuse_output_indices = fuse_output_indices
        self.gc_when_reload = gc_when_reload
        self.group_conv_rewrite = group_conv_rewrite
        self.rewrite_quantizable = rewrite_quantizable
        self.tflite_micro_rewrite = tflite_micro_rewrite
        self.map_bilstm_to_lstm = map_bilstm_to_lstm
        self.float16_quantization = float16_quantization
        self.enable_mtk_ops = enable_mtk_ops
        self.max_transpose_dims = max_transpose_dims
        self.hybrid_conv = hybrid_conv

        if quantize_target_type == 'uint8':
            self.q_type = np.uint8
            if self.strict_symmetric_check:
                log.warning('Symmetric quantized model with uint8 is unsupported in most backends of TFLite')
        elif quantize_target_type == 'int8':
            self.q_type = np.int8
        elif quantize_target_type == 'int16':
            if not self.strict_symmetric_check:
                raise AttributeError('Int16 quantization requires strict_symmetric_check=True')
            self.q_type = np.int16
        else:
            raise AttributeError(f'unknown quantize_target_type: {quantize_target_type}, expected: uint8, int8, int16')

        if quantize_input_output_type is not None:
            assert fuse_quant_dequant, 'Please set fuse_quant_dequant=True, otherwise quantize_input_type is ignored'
            assert quantize_input_output_type in (
                'int8',
                'uint8',
                'int16',
            ), f'unknown quantize_input_output_type: {quantize_input_output_type}, expected: uint8, int8, int16'
            if quantize_input_output_type == 'int16' and quantize_target_type != 'int16':
                raise AttributeError(
                    'quantize_input_output_type == \'int16\' and quantize_target_type != \'int16\' is not supported'
                )
        self.quantize_input_output_type = quantize_input_output_type

        if hybrid_quantize_weight_type is None:
            hybrid_quantize_weight_type = quantize_target_type

        if hybrid_quantize_weight_type == 'uint8':
            if self.hybrid:
                if self.hybrid_per_channel:
                    raise AttributeError('Per-channel kernels supports int8 only')
                log.warning(
                    'Unless you are using legacy TFLite (<1.14), please set quantize_target_type to int8 instead'
                )
            self.hybrid_q_type = np.uint8
        elif hybrid_quantize_weight_type == 'int8':
            self.hybrid_q_type = np.int8
        elif hybrid_quantize_weight_type == 'int16':
            raise AttributeError('Hybrid kernels supports int8 and uint8 only')

        if dump_config_path and not dump_jit_model_path:
            raise AssertionError("when dump_config_path is set, dump_jit_model_path is required to be set")

        self.input_offset = 1

    def init_jit_graph(self):
        # Multi-GPU modules doesn't support JIT tracing
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.model = self.model.module

        if not isinstance(self.model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
            if hasattr(self.model, 'cpu'):
                self.model.cpu()

            if hasattr(self.model, 'eval'):
                self.model.eval()

            with torch.no_grad():
                script = torch.jit.trace(self.model, self.dummy_input)

                # Remove reference to original model to save memory
                self.model = None

                # Have to save it once, otherwise something weird happens
                if self.dump_jit_model_path is None:
                    with io.BytesIO() as f:
                        torch.jit.save(script, f)
                        f.seek(0)
                        script = torch.jit.load(f)
                else:
                    jit_model_dir = os.path.abspath(os.path.dirname(self.dump_jit_model_path))
                    os.makedirs(jit_model_dir, exist_ok=True)
                    torch.jit.save(script, self.dump_jit_model_path)
                    if self.gc_when_reload:
                        import gc

                        script = None
                        gc.collect()

                    script = torch.jit.load(self.dump_jit_model_path)

            self.model = script

        if isinstance(self.model, torch.jit.ScriptFunction):
            self.input_offset = 0

        if self.dump_dummy_input_path is not None:
            dummy_arrs = list(map(lambda x: x.detach().cpu().numpy(), self.dummy_input))
            np.savez(self.dump_dummy_input_path, *dummy_arrs)

        if self.dump_config_path is not None:
            generate_converter_config(
                self.dummy_input,
                [],
                self.input_transpose,
                [],
                self.dump_jit_model_path,
                self.tflite_path,
                self.dump_config_path,
            )

    def init_lowered_module(self):
        assert (
            isinstance(self.model, torch.jit.ScriptFunction)
            or self.model.training is False
            or str(next(self.model.graph.inputs()).type()) == '__torch__.PlaceholderModule'
        ), (
            'Model is in training mode. Please run `model.eval()` before model conversion. If you are passing in a'
            ' TorchScript model, make sure you use `torch.jit.save` to dump the model to disk and then load it using'
            ' `torch.jit.load`.'
        )

        graph = self.model.graph

        # Inline everything
        torch._C._jit_pass_inline(graph)

        # Remove fork/wait nodes
        torch._C._jit_pass_inline_fork_wait(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_lower_all_tuples(graph)

        # we record now record some ops like ones/zeros
        # into a trace where we previously recorded constants
        # use constant prop to maintain our current level of onnx support
        # without implementing symbolics for all of them
        torch._C._jit_pass_constant_propagation(graph)

        # _split_tensor_list_constants(graph, graph)
        # run dce to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_canonicalize_graph_fuser_ops(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_fuse_addmm(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lower_all_tuples(graph)

        self.graph = graph

        log.debug('Lowered graph:')
        log.debug(self.graph)

    def init_input_transpose(self):
        input_transpose = self.input_transpose
        if type(input_transpose) not in (tuple, list):
            input_transpose = [input_transpose] * len(self.dummy_input)
        for i, t in enumerate(self.dummy_input):
            if input_transpose[i] is None:
                if isinstance(t, torch.Tensor):
                    input_transpose[i] = t.dim() == 4
                else:
                    input_transpose[i] = False
        self.input_transpose = input_transpose

    def init_common_graph(self):
        graph_inputs = [x.debugName() for x in list(self.graph.inputs())][self.input_offset :]
        graph_outputs = [x.debugName() for x in list(self.graph.outputs())]
        self.common_graph.inputs.extend(graph_inputs)
        self.common_graph.outputs.extend(graph_outputs)
        self.common_graph.input_transpose.extend(self.input_transpose)
        self.common_graph.output_transpose = self.output_transpose
        tensors = []
        for i, node in enumerate(graph_inputs):
            tensors.append(
                Tensor(
                    self.dummy_input[i],
                    node,
                    has_buffer=False,
                    asymmetric=not self.strict_symmetric_check,
                    q_type=self.q_type,
                )
            )
        self.common_graph.add_nodes(tensors, ExtendedOperator.INPUT_NODE)

    def init_inputs(self):
        graph_inputs = [x.debugName() for x in list(self.graph.inputs())]
        for i, node in enumerate(graph_inputs):
            if self.input_offset > 0 and i == 0:
                self.tensor_map[graph_inputs[i]] = self.model
            else:
                self.tensor_map[graph_inputs[i]] = self.dummy_input[i - self.input_offset]

    def unsupported_operations(self, unique=True) -> typing.List[str]:
        """Returns unsupported operations in the graph"""

        if self.graph is None:
            self.init_lowered_module()

        all_nodes = list(self.graph.nodes())
        ops = []
        for node in all_nodes:
            k = node.kind()
            converter_type = OPERATOR_CONVERTER_DICT.get(k, None)
            if converter_type is None:
                ops.append(k)

        if unique:
            return list(set(ops))
        else:
            return ops

    def init_operations(self):
        log.debug('Initialize operators...')
        node_queue = collections.deque(self.graph.nodes())
        while node_queue:
            node = node_queue.popleft()

            k = node.kind()
            output_tensors = []

            converter_type = OPERATOR_CONVERTER_DICT.get(k, NoTrackOperator)
            converter = converter_type(
                node,
                self.tensor_map,
                not self.strict_symmetric_check,
                self.q_type,
                self.hybrid_q_type,
                self.map_bilstm_to_lstm,
                self.enable_mtk_ops,
                self.hybrid_asymmetric_inputs,
            )
            # Don't track the operator if all the input nodes are not tracked unless it has custom implementation
            # (e.g prim::* ops)
            if converter_type.run == NoTrackOperator.run and converter_type != NoTrackOperator:
                no_track_flag = True
                for n in converter.input_names:
                    if self.common_graph.has_nested_names(n):
                        nested_names = self.common_graph.get_list_expanded_names(n)
                        for x in nested_names:
                            if x in self.common_graph.tensor_map:
                                no_track_flag = False
                                break
                    elif n in self.common_graph.tensor_map:
                        no_track_flag = False
                        break
                if no_track_flag:
                    if converter_type == ATenDequantizeOperator:
                        converter_type = TrackQParamsOperator
                    else:
                        converter_type = NoTrackOperator
                    converter = converter_type(
                        node, self.tensor_map, not self.strict_symmetric_check, self.q_type, self.hybrid_q_type
                    )
            if k != 'prim::Constant':
                log.debug(f'{k} {converter.input_names} -> {converter.output_names} {converter_type.__name__}')
            # Don't fetch attrs and schemas for non-tracking nodes
            if converter_type not in (NoTrackOperator, TrackQParamsOperator):
                try:
                    attrs = converter.fetch_all_attrs(node)
                except StopIteration:
                    attrs = None
                args = converter.fetch_annotated_args(node)
            else:
                attrs = None
                args = None
            converter.parse(node, attrs, args, self.common_graph)
            outputs = converter.output_names
            new_nodes = converter.output_nodes
            if output_tensors is not None:
                output_tensors.extend(converter.get_output_tensors())
            if len(new_nodes) > 0:
                node_queue.extendleft(reversed(new_nodes))

            assert len(output_tensors) == len(outputs)
            for t, name in zip(output_tensors, outputs):
                self.tensor_map[name] = t
                if self.preserve_tensors and isinstance(t, torch.Tensor):
                    self.tensor_map_copies[name] = t.detach().clone()

    def __try_infer_type(self, params):
        try:
            inferred = torch._C._jit_try_infer_type(params)
            if hasattr(inferred, 'type'):
                return inferred.type().annotation_str
        finally:
            return str(inferred)

    def __unpack_params(self, params):
        return NoTrackOperator.unpack_params(None, params)

    def convert(self):
        """Converts the model to the TFLite format

        Raises:
            Exception: If unsupported ops are found, an Exception will be raised
        """
        self.init_input_transpose()
        self.init_jit_graph()
        self.init_lowered_module()
        self.init_common_graph()
        self.init_inputs()
        self.init_operations()

        unsupported_ops = self.unsupported_operations()
        if len(unsupported_ops) > 0:
            log.error(f'Unsupported ops: {", ".join(unsupported_ops)}')
            raise Exception("Cannot continue due to fatal error")
        else:
            optimizer = GraphOptimizer(
                self.common_graph,
                self.optimize,
                self.fuse_quant_dequant,
                self.group_conv_rewrite,
                self.rewrite_quantizable,
                self.tflite_micro_rewrite,
                self.quantize_input_output_type,
                self.fuse_input_indices,
                self.fuse_output_indices,
                self.max_transpose_dims,
            )
            optimizer.optimize()

            self.output_transpose = self.common_graph.output_transpose

            if self.hybrid:
                quantizer = HybridQuantizer(
                    self.common_graph,
                    self.hybrid_asymmetric_inputs,
                    self.hybrid_q_type,
                    self.hybrid_per_channel,
                    self.hybrid_conv,
                )
                quantizer.quantize()
                optimizer.cleanup_dead_nodes()

            if self.float16_quantization:
                quantizer = HalfQuantizer(self.common_graph)
                quantizer.quantize()
                optimizer.cleanup_dead_nodes()

            versioner = OPVersioner(self.common_graph)
            versioner.process()

            self.common_graph.convert(self.tflite_path)

        log.info(f'Generated model saved to {self.tflite_path}')

    def visualize(self, hide_constants=True):
        """Visualize the TinyNeuralNetwork Graph

        Args:
            hide_constants (bool, optional): Hide the constant nodes in the graph. Defaults to True.
        """

        self.common_graph.visualize(hide_constants)

    def get_outputs(self):
        """Returns the output of the model, which is evaluated via tracing nodes one by one"""

        outputs = []
        for name in self.common_graph.outputs:
            outputs.append(self.tensor_map[name])
        return outputs

    def get_value(self, name, default_val=None):
        """Returns the output according to the name of the node. If the name doesn't exist, `default_val` is returned"""

        if self.preserve_tensors:
            val = self.tensor_map_copies.get(name, default_val)
        else:
            val = self.tensor_map.get(name, default_val)

        type_ = self.__try_infer_type(val)
        if type_.endswith('PackedParamsBase'):
            return self.__unpack_params(val)

        return val

    def tensor_names(self) -> typing.List[str]:
        """Returns the all the names of the intermediate tensors

        Returns:
            typing.List[str]: The names of the intermediate tensors
        """

        if self.preserve_tensors:
            return list(self.tensor_map_copies.keys())
        else:
            return list(self.tensor_map.keys())

    def inputs_for_tflite(self) -> typing.List[np.ndarray]:
        """Prepare inputs for the TFLite backend

        Returns:
            typing.List[np.ndarray]: The input tensors
        """

        arrs = []
        for t, trans in zip(self.dummy_input, self.input_transpose):
            arr = t.detach().clone().numpy()
            if trans:
                arr = np.transpose(arr, (0, 2, 3, 1))
            arrs.append(arr)
        return arrs
