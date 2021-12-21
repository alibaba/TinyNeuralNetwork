import collections
import io
import typing
import torch

import numpy as np

from .operators import CommonGraph, ExtendedOperator, GraphOptimizer
from .operators.op_version import OPVersioner
from .operators.tflite import Tensor
from .operators.torch import OPERATOR_CONVERTER_DICT
from .operators.torch.base import NoTrackOperator
from ..util.converter_util import generate_converter_config
from ..util.util import get_logger

log = get_logger(__name__, 'INFO')


class TFLiteConverter(object):
    def __init__(self, model: typing.Union[torch.jit.ScriptFunction, torch.jit.ScriptModule, torch.nn.Module],
                 dummy_input: typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]],
                 tflite_path: str,
                 input_transpose: typing.Optional[typing.Union[bool, typing.Iterable[bool]]] = None,
                 dump_jit_model_path: typing.Optional[str] = None,
                 dump_dummy_input_path: typing.Optional[str] = None,
                 dump_config_path: typing.Optional[str] = None,
                 asymmetric: bool = True,
                 optimize: int = GraphOptimizer.ALL_OPTIMIZE) -> None:
        """ The TFLiteConverter class

        Args:
            model (typing.Union[torch.jit.ScriptFunction, torch.jit.ScriptModule, torch.nn.Module]): The input model (either traced or non-traced)
            dummy_input (typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]]): A viable input to the model
            tflite_path (str): Path to use for exporting
            input_transpose (typing.Optional[typing.Union[bool, typing.Iterable[bool]]], optional): Whether to transpose the input(s). \
                 Defaults to None(True for 4d-input, False otherwise).
            dump_jit_model_path (typing.Optional[str]): The path for dumping the jit model. Defaults to None
            dump_dummy_input_path (typing.Optional[str]): The path for dumping the dummy input. Defaults to None
            dump_config_path (typing.Optional[str]): The path for dumping the json config. Defaults to None
            asymmetric (bool): Asymmetric quantization. Defaults to True
            optimize (int): The level of graph optimization. Defaults to `GraphOptimizer.ALL_OPTIMIZE`
        """

        self.model = model
        self.lower_model = None
        self.graph = None
        self.tensor_map = {}
        self.common_graph = CommonGraph()

        if type(dummy_input) in (tuple, list):
            self.dummy_input = dummy_input
        else:
            self.dummy_input = [dummy_input]

        self.tflite_path = tflite_path
        self.input_transpose = input_transpose
        self.asymmetric = asymmetric

        self.dump_jit_model_path = dump_jit_model_path
        self.dump_dummy_input_path = dump_dummy_input_path
        self.dump_config_path = dump_config_path
        self.optimize = optimize

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

                # Have to save it once, otherwise something weird happens
                if self.dump_jit_model_path is None:
                    with io.BytesIO() as f:
                        torch.jit.save(script, f)
                        f.seek(0)
                        script = torch.jit.load(f)
                else:
                    torch.jit.save(script, self.dump_jit_model_path)
                    script = torch.jit.load(self.dump_jit_model_path)

            self.model = script

        if isinstance(self.model, torch.jit.ScriptFunction):
            self.input_offset = 0

        if self.dump_dummy_input_path is not None:
            dummy_arrs = list(map(lambda x: x.detach().cpu().numpy(), self.dummy_input))
            np.savez(self.dump_dummy_input_path, *dummy_arrs)

        if self.dump_config_path is not None:
            generate_converter_config(self.dummy_input, [], self.input_transpose, [],
                                      self.dump_jit_model_path, self.tflite_path, self.dump_config_path)

    def init_lowered_module(self):
        assert (isinstance(self.model, torch.jit.ScriptFunction) or
                self.model.training is False or
                str(next(self.model.graph.inputs()).type()) == '__torch__.PlaceholderModule'), 'Model is in training model'

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
                input_transpose[i] = t.dim() == 4
        self.input_transpose = input_transpose

    def init_common_graph(self):
        graph_inputs = [x.debugName() for x in list(self.graph.inputs())][self.input_offset:]
        graph_outputs = [x.debugName() for x in list(self.graph.outputs())]
        self.common_graph.inputs.extend(graph_inputs)
        self.common_graph.outputs.extend(graph_outputs)
        self.common_graph.input_transpose.extend(self.input_transpose)
        tensors = []
        for i, node in enumerate(graph_inputs):
            tensors.append(Tensor(self.dummy_input[i], node, has_buffer=False, asymmetric=self.asymmetric))
        self.common_graph.add_nodes(tensors, ExtendedOperator.INPUT_NODE)

    def init_inputs(self):
        graph_inputs = [x.debugName() for x in list(self.graph.inputs())]
        for i, node in enumerate(graph_inputs):
            if self.input_offset > 0 and i == 0:
                self.tensor_map[graph_inputs[i]] = self.model
            else:
                self.tensor_map[graph_inputs[i]] = self.dummy_input[i - self.input_offset]

    def unsupported_operations(self, unique=True) -> typing.List[str]:
        """ Returns unsupported operations in the graph """

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
            converter = converter_type(node, self.tensor_map, self.asymmetric)
            # Don't track the operator if all the input nodes are not tracked unless it has custom implementation (e.g prim::* ops)
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
                    converter_type = NoTrackOperator
                    converter = converter_type(node, self.tensor_map, self.asymmetric)
            if k != 'prim::Constant':
                log.debug(f'{k} {converter.input_names} -> {converter.output_names} {converter_type.__name__}')
            # Don't fetch attrs and schemas for non-tracking nodes
            if converter_type != NoTrackOperator:
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

    def __try_infer_type(self, params):
        inferred = torch._C._jit_try_infer_type(params)
        if hasattr(inferred, 'type'):
            return inferred.type().annotation_str
        else:
            return str(inferred)

    def __unpack_params(self, params):
        return NoTrackOperator.unpack_params(None, params)

    def convert(self):
        """ Converts the model to the TFLite format

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
            optimizer = GraphOptimizer(self.common_graph, self.optimize)
            optimizer.optimize()

            versioner = OPVersioner(self.common_graph)
            versioner.process()

            self.common_graph.convert(self.tflite_path)

        log.info(f'Generated model saved to {self.tflite_path}')

    def visualize(self, hide_constants=True):
        """ Visualize the TinyNeuralNetwork Graph

        Args:
            hide_constants (bool, optional): Hide the constant nodes in the graph. Defaults to True.
        """

        self.common_graph.visualize(hide_constants)

    def get_outputs(self):
        """ Returns the output of the model, which is evaluated via tracing nodes one by one """

        outputs = []
        for name in self.common_graph.outputs:
            outputs.append(self.tensor_map[name])
        return outputs

    def get_value(self, name, default_val=None):
        """ Returns the output according to the name of the node. If the name doesn't exist, `default_val` is returned """

        val = self.tensor_map.get(name, default_val)

        type_ = self.__try_infer_type(val)
        if type_.endswith('PackedParamsBase'):
            return self.__unpack_params(val)

        return val

    def tensor_names(self) -> typing.List[str]:
        """ Returns the all the names of the intermediate tensors

        Returns:
            typing.List[str]: The names of the intermediate tensors
        """

        return list(self.tensor_map.keys())
