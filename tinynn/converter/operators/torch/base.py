from abc import ABC, abstractmethod
from distutils.version import LooseVersion

import ctypes
import inspect
import math
import warnings
import torch
import typing

import numpy as np

from .. import tflite as tfl

from tinynn.util.util import get_logger

log = get_logger(__name__, 'INFO')


class OperatorConverter(ABC):
    def __init__(
        self,
        node,
        tensor_map,
        asymmetric=True,
        q_type=np.uint8,
        hybrid_q_type=np.int8,
        map_bilstm_to_lstm=False,
        enable_mtk_ops=False,
        hybrid_asymmetric_inputs=False,
    ) -> None:
        self.input_names = self.get_input_names(node)
        self.output_names = self.get_output_names(node)
        self.input_tensors = self.get_input_tensors(tensor_map)
        self.output_tensors = []
        self.output_nodes = []
        self.ops = []
        self.attr_count = 0
        self.transform_count = 0
        self.asymmetric = asymmetric
        self.q_type = q_type
        self.hybrid_q_type = hybrid_q_type
        self.map_bilstm_to_lstm = map_bilstm_to_lstm
        self.enable_mtk_ops = enable_mtk_ops
        self.hybrid_asymmetric_inputs = hybrid_asymmetric_inputs

    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        pass

    def get_input_names(self, node):
        return [x.debugName() for x in list(node.inputs())]

    def get_output_names(self, node):
        return [x.debugName() for x in list(node.outputs())]

    def get_input_tensors(self, tensor_map):
        input_tensors = []
        for n in self.input_names:
            if n in tensor_map:
                input_tensors.append(tensor_map[n])
            else:
                raise Exception(f'{n} is not found in the tensor map')
        return input_tensors

    def get_output_tensors(self):
        return self.output_tensors

    def get_ops(self):
        return self.ops

    @staticmethod
    def fetch_all_attrs(node):
        attrs = {}
        for name in node.attributeNames():
            attrs[name] = get_prop_from_node(node, name, return_type=True)
        return attrs

    def fetch_annotated_args(self, node):
        if len(self.input_tensors) == 0:
            return dict()

        k = node.kind()
        if k.startswith('prim::'):
            return dict()

        schemas = torch._C._jit_get_schemas_for_operator(k)
        candidates = []
        for schema in schemas:
            if 'name' in schema.overload_name:
                continue
            if len(schema.arguments) == len(self.input_tensors):
                candidates.append(schema)

        assert len(candidates) > 0, f"Cannot find the schema for {k}({self.output_names[0]})"
        names = (x.name for x in candidates[0].arguments)

        # TODO: Better selection for multiple schemas
        return dict(zip(names, range(len(self.input_tensors))))

    def unimplemented(self, node, attrs, args):
        log.debug(f'node: {node}')
        log.debug('inputs:')
        for name, tensors in zip(self.input_names, self.input_tensors):
            if type(tensors) not in (list, tuple):
                tensors = [tensors]
            for tensor in tensors:
                log.debug(f'name: {name}')
                log.debug(f'tensor: {tensor}')
                if hasattr(tensor, 'shape'):
                    log.debug(f'shape: {tensor.shape}')
                if hasattr(tensor, 'dtype'):
                    log.debug(f'dtype: {tensor.dtype}')
                log.debug('-' * 60)
        log.debug('outputs:')
        for name, tensors in zip(self.output_names, self.output_tensors):
            if type(tensors) not in (list, tuple):
                tensors = [tensors]
            for tensor in tensors:
                log.debug(f'name: {name}')
                log.debug(f'tensor: {tensor}')
                if hasattr(tensor, 'shape'):
                    log.debug(f'shape: {tensor.shape}')
                if hasattr(tensor, 'dtype'):
                    log.debug(f'dtype: {tensor.dtype}')
                log.debug('-' * 60)
        log.debug(f'attrs: {attrs}')
        log.debug(f'args: {args}')
        raise NotImplementedError

    def run(self, node):
        func = torch._C._jit_get_operation(node.kind())
        if isinstance(func, tuple):
            func = func[0]
        with torch.no_grad():
            legacy = True
            if LooseVersion(torch.__version__) >= LooseVersion('1.8.0'):
                try:
                    o = func(*self.input_tensors)
                    legacy = False
                except (TypeError, RuntimeError):
                    pass

            if legacy:
                try:
                    args = self.fetch_annotated_args(node)
                    kwargs = dict(zip(args.keys(), self.input_tensors))
                    o = func(**kwargs)
                except RuntimeError as e:
                    if 'device' in kwargs:
                        kwargs['device'] = 0
                        o = func(**kwargs)
                    else:
                        raise e

        if len(self.output_names) == 1:
            self.output_tensors.append(o)
        else:
            self.output_tensors.extend(o)

    def to_tfl_tensors(
        self, names, tensors, has_buffers=None, graph_converter=None, non_existent_as_buffer=False
    ) -> typing.List[tfl.Tensor]:
        tfl_tensors = []
        if has_buffers is None:
            has_buffers = [None] * len(tensors)
        elif type(has_buffers) == bool:
            has_buffers = [has_buffers] * len(tensors)
        assert len(names) == len(tensors) == len(has_buffers)
        for n, t, b in zip(names, tensors, has_buffers):
            if b is None:
                if graph_converter is not None and n in graph_converter.tensor_map:
                    t = graph_converter.tensor_map[n]
                else:
                    t = tfl.Tensor(
                        t, n, has_buffer=non_existent_as_buffer, asymmetric=self.asymmetric, q_type=self.q_type
                    )
            else:
                t = tfl.Tensor(t, n, has_buffer=b, asymmetric=self.asymmetric, q_type=self.q_type)
            tfl_tensors.append(t)
        return tfl_tensors

    def find_or_create_input(self, idx, graph_converter):
        name = self.input_names[idx]
        if name in graph_converter.tensor_map:
            return graph_converter.tensor_map[name]

        # assert has_buffer, 'only tensors with has_buffer=True can be created at this time,' + \
        #     ' when you encounter this message, it means some ops in the computation graph is not supported'''

        tensor = self.input_tensors[idx]
        return tfl.Tensor(tensor, name, has_buffer=True, asymmetric=self.asymmetric, q_type=self.q_type)

    def get_unique_attr_name(self):
        if self.attr_count == 0:
            name = self.output_names[0] + '_attr'
        else:
            name = self.output_names[0] + f'_attr_{self.attr_count}'
        self.attr_count += 1
        return name

    def get_unique_transform_name(self):
        if self.transform_count == 0:
            name = self.output_names[0] + '_transform'
        else:
            name = self.output_names[0] + f'_transform_{self.transform_count}'
        self.transform_count += 1
        return name

    def create_transform_tensor(self, tensor, name=None, quantization=None):
        if name is None:
            name = self.get_unique_transform_name()
        return tfl.Tensor(
            tensor, name, has_buffer=False, quantization=quantization, asymmetric=self.asymmetric, q_type=self.q_type
        )

    def create_attr_tensor(self, tensor, name=None, hybrid=False):
        if name is None:
            name = self.get_unique_attr_name()

        if hybrid:
            q_type = np.int8
        else:
            q_type = self.q_type

        tensor = tfl.Tensor(tensor, name, has_buffer=True, asymmetric=self.asymmetric, q_type=q_type)

        if hybrid and self.hybrid_q_type == np.uint8:
            tensor.reinterpret_as(self.hybrid_q_type)

        return tensor

    def unpack_params(self, params):
        result = {}
        for method in params._method_names():
            if not (method.startswith('__') and method.endswith('__')):
                result[method] = getattr(params, method)()
        state = params.__getstate__()
        return result, state

    def rescale_weight_scale_for_qnnpack(
        self, input_tensor: tfl.Tensor, weight_tensor: tfl.Tensor, output_tensor: tfl.Tensor
    ):
        updated = False
        orig_scale = weight_tensor.quantization.scale
        while True:
            input_product_scale = input_tensor.quantization.scale * weight_tensor.quantization.scale
            scale = input_product_scale / output_tensor.quantization.scale
            shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23)
            if shift >= 32:
                updated = True
                weight_tensor.quantization.scale *= 10
            else:
                break
        if updated:
            cur_scale = weight_tensor.quantization.scale
            log.info(f'rescale quantized weight of {weight_tensor.name}: {orig_scale:.8f}->{cur_scale:.8f}')

    def quantize(self, tensor, scale, zero_point, dtype=torch.uint8, dim=None):
        if isinstance(scale, list):
            scale = torch.tensor(scale)
        if isinstance(zero_point, list):
            zero_point = torch.tensor(zero_point)
        q_tensor = torch.round(tensor.detach() / scale + zero_point)
        type_info = torch.iinfo(dtype)
        if (q_tensor > type_info.max).any():
            warnings.warn('Overflow while quantizing the tensor')
        if (q_tensor < type_info.min).any():
            warnings.warn('Underflow while quantizing the tensor')
        q_tensor = q_tensor.to(dtype=dtype)
        if isinstance(scale, torch.Tensor):
            scale = scale.tolist()
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.tolist()
        return tfl.FakeQuantTensor(q_tensor, scale, zero_point, dim)

    def passthrough(self, graph_converter):
        assert len(self.input_tensors) >= len(self.output_tensors)

        for i in range(len(self.output_tensors)):
            input_tensor = self.input_tensors[i]
            inputs = [self.find_or_create_input(i, graph_converter), self.create_attr_tensor(input_tensor.shape)]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(tfl.ReshapeOperator(inputs, outputs, input_tensor.shape))

    def elementwise_unary(self, converter_class, graph_converter, *args, **kwargs):
        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if inputs[0].buffer is None:
            graph_converter.add_operator(converter_class(inputs, outputs, *args, **kwargs))

    def elementwise_binary(self, converter_class, graph_converter, autocast, *args, **kwargs):
        if autocast:
            result_dtype = torch.promote_types(self.input_tensors[0].dtype, self.input_tensors[1].dtype)
            for i in range(2):
                t = self.input_tensors[i]
                if result_dtype != t.dtype:
                    casted = t.clone().to(dtype=result_dtype)
                    inp_t = self.find_or_create_input(i, graph_converter)
                    if inp_t.buffer is None:
                        new_inp = self.create_transform_tensor(casted)
                        graph_converter.add_operator(
                            tfl.CastOperator(
                                [inp_t],
                                [new_inp],
                                tfl.torch_tflite_dtype_mappings[t.dtype],
                                tfl.torch_tflite_dtype_mappings[result_dtype],
                            )
                        )
                        self.input_names[i] = new_inp.name
                    self.input_tensors[i] = casted

        inputs = [self.find_or_create_input(i, graph_converter) for i in range(2)]
        if not all((t.buffer is not None for t in inputs)):
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(converter_class(inputs, outputs, *args, **kwargs))

    def reshape(self, graph_converter):
        new_shape = np.array(self.output_tensors[0].shape, dtype='int32')
        inputs = [self.find_or_create_input(0, graph_converter), self.create_attr_tensor(new_shape)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.ReshapeOperator(inputs, outputs, new_shape))

    def wrap_ops_with_dequant_quants(
        self, ops: typing.List[tfl.BaseOperator], input_idx: int = 0, output_idx: int = 0
    ) -> typing.List[tfl.BaseOperator]:
        orig_input = ops[0].inputs[input_idx]
        orig_output = ops[-1].outputs[output_idx]

        new_input = self.create_transform_tensor(orig_input.tensor.astype('float32'))
        new_output = self.create_transform_tensor(orig_output.tensor.astype('float32'))

        dequant_op = tfl.DequantizeOperator([orig_input], [new_input])
        quant_op = tfl.QuantizeOperator([new_output], [orig_output])

        ops[0].inputs[input_idx] = new_input
        ops[-1].outputs[output_idx] = new_output

        return [dequant_op] + ops + [quant_op]

    def wrap_ops_with_2d_3d_reshapes(
        self, ops: typing.List[tfl.BaseOperator], input_idx: int = 0, output_idx: int = 0
    ) -> typing.List[tfl.BaseOperator]:
        orig_input = ops[0].inputs[input_idx]
        orig_output = ops[-1].outputs[output_idx]

        input_shape = np.array(orig_input.tensor.shape[1:], dtype='int32')
        output_shape = np.array(orig_output.tensor.shape, dtype='int32')

        input_shape_tensor = self.create_attr_tensor(input_shape)
        output_shape_tensor = self.create_attr_tensor(output_shape)

        new_input = self.create_transform_tensor(
            orig_input.tensor.reshape(input_shape), quantization=orig_input.quantization
        )
        new_output = self.create_transform_tensor(
            orig_output.tensor.reshape(output_shape[1:]), quantization=orig_output.quantization
        )

        input_reshape_op = tfl.ReshapeOperator([orig_input, input_shape_tensor], [new_input], input_shape)
        output_reshape_op = tfl.ReshapeOperator([new_output, output_shape_tensor], [orig_output], output_shape)

        input_reshape_op.extra_hints['direction'] = 'up'
        output_reshape_op.extra_hints['direction'] = 'down'

        ops[0].inputs[input_idx] = new_input
        ops[-1].outputs[output_idx] = new_output

        return [input_reshape_op] + ops + [output_reshape_op]

    def wrap_ops_with_nhwc_nchw_transposes(
        self, ops: typing.List[tfl.BaseOperator], input_idx: int = 0, output_idx: int = 0
    ) -> typing.List[tfl.BaseOperator]:
        orig_input = ops[0].inputs[input_idx]
        orig_output = ops[-1].outputs[output_idx]

        nhwc2nchw_perm = np.array([0, 3, 1, 2], dtype='int32')
        nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')

        nhwc2nchw_perm_tensor = self.create_attr_tensor(nhwc2nchw_perm)
        nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)

        new_input = self.create_transform_tensor(
            np.transpose(orig_input.tensor, nchw2nhwc_perm), quantization=orig_input.quantization
        )
        new_output = self.create_transform_tensor(
            np.transpose(orig_output.tensor, nchw2nhwc_perm), quantization=orig_output.quantization
        )

        nchw2nhwc_transpose = tfl.TransposeOperator([orig_input, nchw2nhwc_perm_tensor], [new_input])
        nhwc2nchw_transpose = tfl.TransposeOperator([new_output, nhwc2nchw_perm_tensor], [orig_output])

        nchw2nhwc_transpose.extra_hints['direction'] = 'up'
        nhwc2nchw_transpose.extra_hints['direction'] = 'down'

        ops[0].inputs[input_idx] = new_input
        ops[-1].outputs[output_idx] = new_output

        return [nchw2nhwc_transpose] + ops + [nhwc2nchw_transpose]

    def wrap_ops_with_last_dim_transposes(
        self, ops: typing.List[tfl.BaseOperator], dim: int, input_idx: int = 0, output_idx: int = 0
    ) -> typing.List[tfl.BaseOperator]:
        orig_input = ops[0].inputs[input_idx]
        orig_output = ops[-1].outputs[output_idx]

        assert len(orig_input.shape) == len(orig_output.shape), "Numbers of dimensions mismatch"

        n_dim = len(orig_input.shape)
        if n_dim == dim:
            return ops

        last_dim_perm = np.array([i for i in range(n_dim) if i != dim] + [dim], dtype='int32')
        rev_last_dim_perm = np.argsort(last_dim_perm).astype('int32')

        last_dim_perm_tensor = self.create_attr_tensor(last_dim_perm)
        rev_last_dim_perm_tensor = self.create_attr_tensor(rev_last_dim_perm)

        new_input = self.create_transform_tensor(
            np.transpose(orig_input.tensor, last_dim_perm), quantization=orig_input.quantization
        )
        new_output = self.create_transform_tensor(
            np.transpose(orig_output.tensor, last_dim_perm), quantization=orig_output.quantization
        )

        last_dim_transpose = tfl.TransposeOperator([orig_input, last_dim_perm_tensor], [new_input])
        rev_last_dim_transpose = tfl.TransposeOperator([new_output, rev_last_dim_perm_tensor], [orig_output])

        last_dim_transpose.extra_hints['direction'] = 'up'
        rev_last_dim_transpose.extra_hints['direction'] = 'down'

        ops[0].inputs[input_idx] = new_input
        ops[-1].outputs[output_idx] = new_output

        return [last_dim_transpose] + ops + [rev_last_dim_transpose]

    def handle_padding(self, pad_h, pad_w, pad_op_index, ops, ceil_mode=False):
        fill_nan = False
        if ceil_mode:
            input_tensor = ops[0].inputs[0]
            kernel_size = [ops[1].filterHeight, ops[1].filterWidth]
            stride = [ops[1].strideH, ops[1].strideW]
            padding = [pad_h, pad_w]

            input_size = [input_tensor.shape[2], input_tensor.shape[3]]

            if not all((i + 2 * p - k) % s == 0 for i, p, k, s in zip(input_size, padding, kernel_size, stride)):
                assert type(ops[1]) == tfl.MaxPool2dOperator, 'ceil_mode=True for AvgPool not supported'
                fill_nan = True
                ceil_pad = get_pool_ceil_padding(input_tensor, kernel_size, stride, padding)
                ceil_pad = list(np.add(ceil_pad, padding))

        if pad_h + pad_w > 0:
            pad = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
            pad_tensor = self.create_attr_tensor(np.array(pad, dtype='int32'))

            pad_input = ops[pad_op_index - 1].outputs[0]

            inputs = [pad_input, pad_tensor]
            if type(ops[1]) == tfl.MaxPool2dOperator:
                constant_tensor = self.get_minimum_constant(pad_input)
                inputs.append(constant_tensor)
                pad_array = np.pad(pad_input.tensor, pad, constant_values=constant_tensor.tensor[0])
            else:
                pad_array = np.pad(pad_input.tensor, pad)

            pad_out = self.create_transform_tensor(pad_array, quantization=pad_input.quantization)
            ops[pad_op_index].inputs[0] = pad_out

            if len(inputs) > 2:
                pad_op = tfl.Padv2Operator(inputs, [pad_out])
            else:
                pad_op = tfl.PadOperator(inputs, [pad_out])
            ops.insert(pad_op_index, pad_op)

        if fill_nan:
            fill_nan_index = pad_op_index + 1 if pad_h + pad_w > 0 else pad_op_index
            pad = [[0, 0], [0, ceil_pad[0]], [0, ceil_pad[1]], [0, 0]]
            pad_tensor = self.create_attr_tensor(np.array(pad, dtype='int32'))
            pad_input = ops[fill_nan_index - 1].outputs[0]
            constant_tensor = self.get_minimum_constant(pad_input)

            pad_array = np.pad(pad_input.tensor, pad, constant_values=constant_tensor.tensor[0])
            pad_out = self.create_transform_tensor(pad_array, quantization=pad_input.quantization)
            ops[fill_nan_index].inputs[0] = pad_out

            pad_op = tfl.Padv2Operator([pad_input, pad_tensor, constant_tensor], [pad_out])
            ops.insert(fill_nan_index, pad_op)

    def get_minimum_constant(self, ref_tensor):
        if ref_tensor.quantization is not None:
            if self.q_type == np.uint8:
                nan = 0
                constant_arr = tfl.FakeQuantTensor(
                    np.zeros(1, dtype=ref_tensor.dtype),
                    ref_tensor.quantization.scale,
                    ref_tensor.quantization.zero_point,
                )
            else:
                nan = -128
                constant_arr = tfl.FakeQuantTensor(
                    np.array([-128], dtype=ref_tensor.dtype),
                    ref_tensor.quantization.scale,
                    ref_tensor.quantization.zero_point,
                )
        else:
            nan = np.finfo(np.float32).min
            constant_arr = np.array([nan], dtype='float32')

        constant_tensor = self.create_attr_tensor(constant_arr)
        return constant_tensor

    def handle_reduce(self, converter_class, input_args, graph_converter, transpose_opt, *args, **kwargs):
        input_tensor = self.find_or_create_input(0, graph_converter)

        if 'dim' in input_args and 'keepdim' in input_args:
            dims, keep_dim = self.input_tensors[1:3]
            if type(dims) not in (list, tuple):
                dims = [dims]
            if len(dims) == 0:
                dims = list(range(input_tensor.tensor.ndim))
                self.output_tensors[0] = self.output_tensors[0].view(1)
        else:
            dims = list(range(input_tensor.tensor.ndim))
            keep_dim = False
            self.output_tensors[0] = self.output_tensors[0].view(1)

        for idx, dim in enumerate(dims):
            if dim < 0:
                dims[idx] += input_tensor.tensor.ndim

        ops = []
        transpose = False

        if transpose_opt:
            # For some ops the codepath is optimized for nhwc.
            # For example, for tfl.Mean, if it is a pooling 2d op, consider wrapping it with transposes
            if len(input_tensor.shape) == 4 and keep_dim in (1, True):
                if dims == [2, 3]:
                    dims = [1, 2]
                    transpose = True
                elif dims == [3, 2]:
                    dims = [2, 1]
                    transpose = True

        dim_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))

        inputs = [input_tensor, dim_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if len(outputs) > 1:
            log.warning(
                'Reduce ops like `torch.min` have multiple outputs. However, only the first '
                'output will be preserved in our converter. If you need that tensor, please '
                'use the `torch.argmin` instead.'
            )
            outputs = outputs[:1]

        if (
            hasattr(converter_class, '__init__')
            and 'keepDims' in inspect.signature(converter_class.__init__).parameters
        ):
            ops.append(converter_class(inputs, outputs, keep_dim, *args, **kwargs))
        else:
            if keep_dim:
                output_tensor = outputs[0]
                transform = self.create_transform_tensor(np.squeeze(output_tensor.tensor, tuple(dims)))
                ops.append(converter_class(inputs, [transform], *args, **kwargs))

                shape_tensor = self.create_attr_tensor(np.array(output_tensor.shape, dtype='int32'))
                ops.append(tfl.ReshapeOperator([transform, shape_tensor], [output_tensor], shape_tensor.tensor))
            else:
                ops.append(converter_class(inputs, outputs, *args, **kwargs))

        if transpose:
            if keep_dim:
                ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)
            else:
                orig_input = ops[0].inputs[0]

                nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')
                nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)

                new_input = self.create_transform_tensor(
                    np.transpose(orig_input.tensor, nchw2nhwc_perm), quantization=orig_input.quantization
                )

                nchw2nhwc_transpose = tfl.TransposeOperator([orig_input, nchw2nhwc_perm_tensor], [new_input])

                ops[0].inputs[0] = new_input
                ops.insert(0, nchw2nhwc_transpose)

        for op in ops:
            graph_converter.add_operator(op)

    def quantize_scalar_tensor(self, tensor: torch.Tensor):
        assert tensor.numel() == 1
        assert tensor.dtype == torch.float32
        if not tensor.is_nonzero():
            if self.q_type in (np.uint8, np.int16):
                return torch.quantize_per_tensor(tensor, 0.5, 128, torch.quint8)
            elif self.q_type == np.int8:
                return torch.quantize_per_tensor(tensor, 0.5, 0, torch.qint8)
        elif (torch.sign(tensor) < 0).all():
            if self.q_type == np.uint8:
                return torch.quantize_per_tensor(tensor, -tensor[0] / 127, 255, torch.quint8)
            elif self.q_type == np.int8:
                return torch.quantize_per_tensor(tensor, -tensor[0] / 127, 0, torch.qint8)
            elif self.q_type == np.int16:
                return torch.quantize_per_tensor(tensor, -tensor[0] / 127, 128, torch.quint8)
        else:
            if self.q_type == np.uint8:
                return torch.quantize_per_tensor(tensor, tensor[0] / 127, 0, torch.quint8)
            elif self.q_type == np.int8:
                return torch.quantize_per_tensor(tensor, tensor[0] / 127, 0, torch.qint8)
            elif self.q_type == np.int16:
                return torch.quantize_per_tensor(tensor, tensor[0] / 127, 128, torch.quint8)

    def torch_tensor_from_scalar(self, ref_tensor: torch.Tensor, src_tensor: torch.Tensor):
        tgt_tensor = src_tensor
        if type(src_tensor) != torch.Tensor:
            if ref_tensor.is_quantized:
                tgt_tensor = torch.quantize_per_tensor(
                    torch.tensor([src_tensor], dtype=torch.float32),
                    ref_tensor.q_scale(),
                    ref_tensor.q_zero_point(),
                    ref_tensor.dtype,
                )
            else:
                tgt_tensor = torch.tensor([src_tensor], dtype=ref_tensor.dtype)
        return tgt_tensor


def get_prop_from_node(node, prop, assert_type=None, return_type=False):
    output_name = next(node.outputs()).debugName()
    if prop in node.attributeNames():
        vk = node.kindOf(prop)
        if assert_type is not None and vk != assert_type:
            return None

        if vk == 'i':
            v = getattr(node, vk)(prop)
        elif vk == 'f':
            v = getattr(node, vk)(prop)
        elif vk == 's':
            v = getattr(node, vk)(prop)
        elif vk == 't':
            v = getattr(node, vk)(prop)
            if v.dtype == torch.float64:
                log.warning(
                    f'{output_name} is of type float64, which is unsupported in TFLite, trying to downcast to float32'
                )
                v = v.to(dtype=torch.float32)
        elif node.output().type().isSubtypeOf(torch._C.ListType.ofInts()) or node.output().type().isSubtypeOf(
            torch._C.ListType.ofFloats()
        ):
            v = node.output().toIValue()
        elif vk == 'ival':
            v = node.output().toIValue()
        else:
            log.warning(f'Skip unsupported constant generation for {output_name}, type: {vk}')
            raise StopIteration
    else:
        v = None
        vk = None

    if return_type:
        return v, vk
    else:
        return v


def fp32_to_bits(val):
    b = np.float32(val).tobytes()
    return np.frombuffer(b, dtype='uint32')[0]


def get_pool_ceil_padding(input, kernel_size, stride, padding):
    # Copied from the PyTorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py
    sizes = input.shape
    dim = sizes[-len(padding) :] if sizes is not None else None
    ceiled_output_dim = [
        int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i]))) + 1
        for i in range(0, len(padding))
    ]
    # ensure last pooling starts inside
    ceiled_output_dim = [
        ceiled_output_dim[i] - 1
        if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
        else ceiled_output_dim[i]
        for i in range(0, len(ceiled_output_dim))
    ]
    padding_ceil = [
        0
        if (stride[i] == 1)
        else (kernel_size[i] - (dim[i] + 2 * padding[i] - ((ceiled_output_dim[i] - 1) * stride[i] + 1)))
        for i in range(0, len(padding))
    ]
    # ensure padding is not > kernel_size
    padding_ceil = [
        (int(padding_ceil[i]) if padding_ceil[i] < kernel_size[i] - 1 else int(kernel_size[i] - 1))
        if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
        else int(padding_ceil[i])
        for i in range(0, len(padding_ceil))
    ]
    return padding_ceil


class NoTrackOperator(OperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)


class TrackQParamsOperator(OperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        t = self.find_or_create_input(0, graph_converter)
        graph_converter.q_mapping[self.output_names[0]] = t


class PrimOperatorConverter(OperatorConverter):
    # prim::* ops needs custom implementation
    def run(self, node):
        pass
