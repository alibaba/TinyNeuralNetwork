from abc import abstractmethod

from .base import BaseOperator, QuantizationParameters, Tensor
from .custom import MTKTransposeConvOperator
from . import generated_ops as tfl_ops

from ..base import ExtendedOperator
from ...schemas.tflite import schema_generated as tflite

import typing
import torch
import warnings

import numpy as np


class TransformableOperator(BaseOperator):
    def __init__(self, op: int, inputs: typing.List['Tensor'], outputs: typing.List['Tensor'], op_version: int):
        super().__init__(op, inputs, outputs, op_version=op_version)
        self.attr_count = 0
        self.transform_count = 0

    @abstractmethod
    def transform(self):
        pass

    def create_attr_tensor(self, tensor, name=None, quantization=None):
        if name is None:
            if self.attr_count == 0:
                name = self.outputs[0].name + '_te_attr'
            else:
                name = self.outputs[0].name + f'_te_attr_{self.attr_count}'
            self.attr_count += 1
        return Tensor(tensor, name, has_buffer=True, quantization=quantization)

    def create_transform_tensor(self, tensor, name=None, quantization=None):
        if name is None:
            if self.transform_count == 0:
                name = self.outputs[0].name + '_te_transform'
            else:
                name = self.outputs[0].name + f'_te_transform_{self.transform_count}'
            self.transform_count += 1
        return Tensor(tensor, name, has_buffer=False, quantization=quantization)

    def wrap_ops_with_nhwc_nchw_transposes(
        self, ops: typing.List[tfl_ops.BaseOperator], input_idx: int = 0, output_idx: int = 0
    ) -> typing.List[tfl_ops.BaseOperator]:
        orig_input = ops[0].inputs[input_idx]
        orig_output = ops[-1].outputs[output_idx]

        if orig_input.tensor.ndim == 4:
            nhwc2nchw_perm = np.array([0, 3, 1, 2], dtype='int32')
            nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')
        elif orig_input.tensor.ndim == 5:
            nhwc2nchw_perm = np.array([0, 4, 1, 2, 3], dtype='int32')
            nchw2nhwc_perm = np.array([0, 2, 3, 4, 1], dtype='int32')
        else:
            assert False, f'Don\'t know how to wrap tranposes for {orig_input.tensor.ndim}d tensors'

        nhwc2nchw_perm_tensor = self.create_attr_tensor(nhwc2nchw_perm)
        nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)

        new_input = self.create_transform_tensor(
            np.transpose(orig_input.tensor, nchw2nhwc_perm), quantization=orig_input.quantization
        )
        new_output = self.create_transform_tensor(
            np.transpose(orig_output.tensor, nchw2nhwc_perm), quantization=orig_output.quantization
        )

        nchw2nhwc_transpose = tfl_ops.TransposeOperator([orig_input, nchw2nhwc_perm_tensor], [new_input])
        nhwc2nchw_transpose = tfl_ops.TransposeOperator([new_output, nhwc2nchw_perm_tensor], [orig_output])

        nchw2nhwc_transpose.extra_hints['direction'] = 'up'
        nhwc2nchw_transpose.extra_hints['direction'] = 'down'

        ops[0].inputs[input_idx] = new_input
        ops[-1].outputs[output_idx] = new_output

        return [nchw2nhwc_transpose] + ops + [nhwc2nchw_transpose]


class BatchNormOperator(TransformableOperator):
    input_index = 0
    weight_index = 1
    bias_index = 2
    running_mean_index = 3
    running_variance_index = 4

    output_index = 0

    def __init__(
        self,
        inputs: typing.List['Tensor'],
        outputs: typing.List['Tensor'],
        eps: float,
        quantization: typing.Optional[QuantizationParameters] = None,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
    ):
        super().__init__(ExtendedOperator.BATCH_NORM, inputs, outputs, 1)
        self.eps = eps
        self.fusedActivationFunction = fusedActivationFunction

    def transform(self, graph_converter, mapping):
        assert all((x.buffer is not None for x in self.inputs[1:]))

        w, b, mean, var = [
            self.inputs[i]
            for i in (self.weight_index, self.bias_index, self.running_mean_index, self.running_variance_index)
        ]

        inv = 1 / np.sqrt(var.tensor + self.eps)
        new_w = inv * w.tensor
        new_b = b.tensor - mean.tensor * new_w

        inp = self.inputs[0]

        new_shape = [1] + [new_w.shape[0]] + [1] * (inp.tensor.ndim - 2)

        new_w = new_w.reshape(new_shape)
        new_b = new_b.reshape(new_shape)

        weight = self.create_attr_tensor(new_w)
        bias = self.create_attr_tensor(new_b)

        new_inp = inp
        if inp.quantization is not None:
            new_inp = self.create_transform_tensor(inp.tensor)
            graph_converter.add_operator(tfl_ops.DequantizeOperator([inp], [new_inp]))

        mul_out = self.create_transform_tensor(new_inp.tensor * weight.tensor)
        graph_converter.add_operator(tfl_ops.MulOperator([new_inp, weight], [mul_out]))

        if inp.quantization is not None:
            add_out = self.create_transform_tensor(mul_out.tensor + bias.tensor)
        else:
            add_out = self.outputs[self.output_index]

        graph_converter.add_operator(
            tfl_ops.AddOperator([mul_out, bias], [add_out], fusedActivationFunction=self.fusedActivationFunction),
            transform=True,
        )

        if inp.quantization is not None:
            quant_out = self.outputs[self.output_index]
            graph_converter.add_operator(tfl_ops.QuantizeOperator([add_out], [quant_out]), transform=True)

        graph_converter.try_restore_edges(mapping)


class GenericConvOperator(TransformableOperator):
    input_index = 0
    weight_index = 1
    bias_index = 2

    output_index = 0

    stride: typing.List[int]
    padding: typing.List[int]
    dilation: typing.List[int]
    transpose: bool
    output_padding: typing.List[int]
    groups: int

    fusedActivationFunction: tflite.ActivationFunctionType

    def __init__(
        self,
        inputs: typing.List['Tensor'],
        outputs: typing.List['Tensor'],
        stride: typing.List[int],
        padding: typing.List[int],
        dialation: typing.List[int],
        output_padding: typing.List[int],
        groups: int,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
    ):
        super().__init__(ExtendedOperator.GENERIC_CONV, inputs, outputs, 1)
        self.stride = stride
        self.padding = padding
        self.dilation = dialation
        self.output_padding = output_padding
        self.groups = groups

        self.fusedActivationFunction = fusedActivationFunction

    def transform(self, graph_converter, mapping):
        input_tensor = self.inputs[0]
        weight_tensor = self.inputs[1]

        input_dim = len(input_tensor.shape)
        weight_dim = len(weight_tensor.shape)

        prev_ops = []
        next_ops = []

        if weight_dim == 3 or input_dim == 3:
            reshape_input_size = 1
            reshape_output_size = 1
            if weight_dim == 3:
                self.stride.insert(0, 1)
                self.padding.insert(0, 0)
                self.dilation.insert(0, 1)
                self.output_padding.insert(0, 0)
                reshape_input_size = 2

            reshape_outputs = [
                self.create_transform_tensor(
                    np.expand_dims(t.tensor, 2),
                    name=f'{self.outputs[0].name}_{t.name}_4d_input',
                    quantization=t.quantization,
                )
                for t in self.inputs[:reshape_input_size]
            ]
            reshape_attrs = [self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in reshape_outputs]
            reshape_ops = [
                tfl_ops.ReshapeOperator([old, attr], [new], attr.tensor)
                for old, new, attr in zip(self.inputs[:reshape_input_size], reshape_outputs, reshape_attrs)
            ]

            for op in reshape_ops:
                op.extra_hints['direction'] = 'up'

            prev_ops.extend(reshape_ops)

            conv_outputs = [
                self.create_transform_tensor(
                    np.expand_dims(self.outputs[i].tensor, 2),
                    name=f'{self.outputs[i].name}_4d_output',
                    quantization=self.outputs[i].quantization,
                )
                for i in range(reshape_output_size)
            ]
            conv_attrs = [
                self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in self.outputs[:reshape_output_size]
            ]
            conv_ops = [
                tfl_ops.ReshapeOperator([old, attr], [new], attr.tensor)
                for old, new, attr in zip(conv_outputs, self.outputs[:reshape_output_size], conv_attrs)
            ]

            for op in conv_ops:
                op.extra_hints['direction'] = 'down'

            next_ops.extend(conv_ops)

            self.inputs = reshape_outputs + self.inputs[reshape_input_size:]
            self.outputs = conv_outputs + self.outputs[reshape_output_size:]

            weight_tensor = self.inputs[1]
        elif weight_dim not in (4, 5):
            assert False, "Only Conv[Transpose]1d/2d/3d is supported"

        if weight_tensor.shape[1] == 1 and weight_tensor.shape[0] == self.groups:
            if weight_dim in (3, 4):
                conv_op = tfl_ops.DepthwiseConv2dOperator(
                    self.inputs,
                    self.outputs,
                    strideH=self.stride[0],
                    strideW=self.stride[1],
                    depthMultiplier=1,
                    dilationHFactor=self.dilation[0],
                    dilationWFactor=self.dilation[1],
                    fusedActivationFunction=self.fusedActivationFunction,
                    padding=tflite.Padding.VALID,
                )
            else:
                assert False, "Only DepthwiseConv1d/2d is supported"
        else:
            if input_tensor.shape[1] != weight_tensor.shape[1]:
                warnings.warn(
                    'Group conv is not supported if official tflite interpreter is used. If that is the case for you,'
                    ' plese pass in `group_conv_rewrite=True`. If you want to run the model with TFLite micro, then you'
                    ' may also need to pass in `tflite_micro_rewrite=True`'
                )
            if weight_dim in (3, 4):
                conv_op = tfl_ops.Conv2dOperator(
                    self.inputs,
                    self.outputs,
                    strideH=self.stride[0],
                    strideW=self.stride[1],
                    dilationHFactor=self.dilation[0],
                    dilationWFactor=self.dilation[1],
                    fusedActivationFunction=self.fusedActivationFunction,
                    padding=tflite.Padding.VALID,
                )
            else:
                conv_op = tfl_ops.Conv3dOperator(
                    self.inputs,
                    self.outputs,
                    strideD=self.stride[0],
                    strideH=self.stride[1],
                    strideW=self.stride[2],
                    dilationDFactor=self.dilation[0],
                    dilationHFactor=self.dilation[1],
                    dilationWFactor=self.dilation[2],
                    fusedActivationFunction=self.fusedActivationFunction,
                    padding=tflite.Padding.VALID,
                )

        ops = self.wrap_ops_with_nhwc_nchw_transposes([conv_op])
        conv_op = ops[1]

        # Pad handling
        if sum(self.padding) > 0:
            if weight_dim in (3, 4):
                pad_h = self.padding[0]
                pad_w = self.padding[1]

                pad = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
            else:
                pad_d = self.padding[0]
                pad_h = self.padding[1]
                pad_w = self.padding[2]

                pad = [[0, 0], [pad_d, pad_d], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]

            pad_tensor = self.create_attr_tensor(np.array(pad, dtype='int32'))

            pad_input = ops[0].outputs[0]
            pad_array = np.pad(pad_input.tensor, pad)
            pad_out = self.create_transform_tensor(pad_array, quantization=pad_input.quantization)
            ops[1].inputs[0] = pad_out

            pad_op = tfl_ops.PadOperator([pad_input, pad_tensor], [pad_out])
            ops.insert(1, pad_op)

        # Weight handling
        weight = conv_op.inputs[1]
        if conv_op.op.code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
            nchw2chwn_perm = np.array([1, 2, 3, 0], dtype='int32')
            nchw2chwn_perm_tensor = self.create_attr_tensor(nchw2chwn_perm)
            weight_q = weight.quantization
            if weight_q is not None and weight_q.dim is not None:
                new_dim = np.nonzero(nchw2chwn_perm == weight_q.dim)[0][0]
                weight_q = QuantizationParameters(weight_q.scale, weight_q.zero_point, new_dim)
            reordered_weight = self.create_transform_tensor(
                np.transpose(weight.tensor, nchw2chwn_perm), quantization=weight_q
            )
            conv_op.inputs[1] = reordered_weight
            reorder_op = tfl_ops.TransposeOperator([weight, nchw2chwn_perm_tensor], [reordered_weight])
        else:
            if weight_dim in (3, 4):
                nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')
                nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)
            else:
                nchw2nhwc_perm = np.array([2, 3, 4, 1, 0], dtype='int32')
                nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)
            weight_q = weight.quantization
            if weight_q is not None and weight_q.dim is not None:
                new_dim = np.nonzero(nchw2nhwc_perm == weight_q.dim)[0][0]
                weight_q = QuantizationParameters(weight_q.scale, weight_q.zero_point, new_dim)
            reordered_weight = self.create_transform_tensor(
                np.transpose(weight.tensor, nchw2nhwc_perm), quantization=weight_q
            )
            conv_op.inputs[1] = reordered_weight
            reorder_op = tfl_ops.TransposeOperator([weight, nchw2nhwc_perm_tensor], [reordered_weight])
        ops.insert(1, reorder_op)

        # Bias handling
        kernel_num = self.inputs[1].shape[0]
        if conv_op.op.code in (tflite.BuiltinOperator.DEPTHWISE_CONV_2D, tflite.BuiltinOperator.CONV_3D):
            kernel_num = self.inputs[1].shape[-1]

        if len(conv_op.inputs) == 2 or conv_op.inputs[2] is None:
            if conv_op.inputs[0].dtype == np.dtype('float32'):
                bias = np.zeros((kernel_num,), dtype='float32')
                q_args = None
            else:
                bias = np.zeros((kernel_num,), dtype='int32')

                per_tensor = weight_tensor.quantization.dim is None

                # Bias handling
                if per_tensor:
                    bias_scale = input_tensor.quantization.scale * weight_tensor.quantization.scale
                    bias_zero_point = 0
                    bias_dim = None
                else:
                    bias_scale = [input_tensor.quantization.scale * s for s in weight_tensor.quantization.scale]
                    bias_zero_point = [0] * len(bias_scale)
                    bias_dim = 0

                q_args = QuantizationParameters(bias_scale, bias_zero_point, bias_dim)

            conv_op.inputs.append(self.create_attr_tensor(bias, quantization=q_args))
        elif conv_op.inputs[2].shape[0] != kernel_num and conv_op.inputs[2].shape[0] == 1:
            if conv_op.inputs[0].dtype == np.float32:
                bias = torch.tensor([conv_op.inputs[2][0]] * kernel_num, dtype='float32')
            else:
                bias = torch.tensor([conv_op.inputs[2][0]] * kernel_num, dtype='int32')

            conv_op.inputs[2] = self.create_attr_tensor(bias)

        ops = prev_ops + ops + next_ops

        for op in ops:
            graph_converter.add_operator(op, transform=True)

        graph_converter.try_restore_edges(mapping)

        for op in ops[:-1]:
            output_name = op.outputs[0].name
            node_name = graph_converter.tensor_node_map[output_name]
            node = graph_converter.graph.vs.find(name=node_name)
            assert node.outdegree() > 0, (
                'The following node should be a part of the transformable node,                 but the outdegree of'
                f' it is zero. {node}'
            )
            next_node = graph_converter.graph.vs[node.out_edges()[0].target]
            assert next_node['node_type'] != ExtendedOperator.CONSTANT_NODE


class GenericTransposeConvOperator(TransformableOperator):
    input_index = 0
    weight_index = 1
    bias_index = 2

    output_index = 0

    stride: typing.List[int]
    padding: typing.List[int]
    dilation: typing.List[int]
    transpose: bool
    output_padding: typing.List[int]
    groups: int

    enable_mtk_ops: bool

    def __init__(
        self,
        inputs: typing.List['Tensor'],
        outputs: typing.List['Tensor'],
        stride: typing.List[int],
        padding: typing.List[int],
        dilation: typing.List[int],
        output_padding: typing.List[int],
        groups: int,
        enable_mtk_ops: bool = False,
    ):
        super().__init__(ExtendedOperator.GENERIC_DECONV, inputs, outputs, 1)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.enable_mtk_ops = enable_mtk_ops

    def transform(self, graph_converter, mapping):
        input_tensor = self.inputs[0]
        weight_tensor = self.inputs[1]
        output_tensor = self.outputs[0]

        input_dim = len(input_tensor.shape)
        weight_dim = len(weight_tensor.shape)

        prev_ops = []
        next_ops = []

        if weight_dim == 3 or input_dim == 3:
            self.stride.insert(0, 1)
            self.padding.insert(0, 0)
            self.dilation.insert(0, 1)
            self.output_padding.insert(0, 0)

            reshape_outputs = [
                self.create_transform_tensor(
                    np.expand_dims(t.tensor, 2),
                    name=f'{self.outputs[0].name}_{t.name}_4d_input',
                    quantization=t.quantization,
                )
                for t in self.inputs[:2]
            ]
            reshape_attrs = [self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in reshape_outputs]
            reshape_ops = [
                tfl_ops.ReshapeOperator([old, attr], [new], attr.tensor)
                for old, new, attr in zip(self.inputs[:2], reshape_outputs, reshape_attrs)
            ]

            for op in reshape_ops:
                op.extra_hints['direction'] = 'up'

            if weight_dim == 3 and input_dim == 3:
                prev_ops.extend(reshape_ops)
            elif weight_dim == 3:
                prev_ops.append(reshape_ops[1])
            else:
                prev_ops.append(reshape_ops[0])

            conv_outputs = [
                self.create_transform_tensor(
                    np.expand_dims(self.outputs[0].tensor, 2),
                    name=f'{self.outputs[0].name}_4d_output',
                    quantization=self.outputs[0].quantization,
                )
            ]
            conv_attrs = [self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in self.outputs[:1]]
            conv_ops = [
                tfl_ops.ReshapeOperator([old, attr], [new], attr.tensor)
                for old, new, attr in zip(conv_outputs, self.outputs[:1], conv_attrs)
            ]

            for op in conv_ops:
                op.extra_hints['direction'] = 'down'

            next_ops.extend(conv_ops)

            if weight_dim == 3 and input_dim == 3:
                self.inputs = reshape_outputs + self.inputs[2:]
            elif weight_dim == 3:
                self.inputs = self.inputs[0:1] + reshape_outputs[1:2] + self.inputs[1:]
            else:
                self.inputs = reshape_outputs[0:1] + self.inputs[1:]
            self.outputs = conv_outputs + self.outputs[1:]

            weight_tensor = self.inputs[1]
        elif weight_dim not in (4, 5):
            assert False, "Only Conv[Transpose]1d/2d/3d is supported"

        if output_tensor.shape[1] != weight_tensor.shape[1]:
            warnings.warn(
                'Group transposed conv is not supported if official tflite interpreter is used. If that is the case'
                ' for you, plese pass in `group_conv_rewrite=True`. If you want to run the model with TFLite micro,'
                ' then you may also need to pass in `tflite_micro_rewrite=True`'
            )

        if weight_dim in (3, 4):
            assert all((x == 1 for x in self.dilation)), "Only dilation=1 is supported for conv_transpose2d"
            if self.enable_mtk_ops:
                conv_op = MTKTransposeConvOperator(
                    self.inputs[:2][::-1],
                    self.outputs,
                    depth_multiplier=1,
                    dilation_height_factor=self.dilation[0],
                    dilation_width_factor=self.dilation[1],
                    padding_type=tflite.Padding.VALID,
                    stride_height=self.stride[0],
                    stride_width=self.stride[1],
                )
            else:
                conv_op = tfl_ops.TransposeConvOperator(
                    self.inputs[:2][::-1],
                    self.outputs,
                    strideH=self.stride[0],
                    strideW=self.stride[1],
                    padding=tflite.Padding.VALID,
                )
        else:
            conv_op = tfl_ops.Conv3dTransposeOperator(
                self.inputs[:2][::-1],
                self.outputs,
                strideD=self.stride[0],
                strideH=self.stride[1],
                strideW=self.stride[2],
                dilationDFactor=self.dilation[0],
                dilationHFactor=self.dilation[1],
                dilationWFactor=self.dilation[2],
                padding=tflite.Padding.VALID,
            )

        ops = self.wrap_ops_with_nhwc_nchw_transposes([conv_op], input_idx=1)

        # Pad handling
        output_shape = conv_op.outputs[0].shape
        if sum(self.padding) > 0:
            if weight_dim in (3, 4):
                pad_h = self.padding[0]
                pad_w = self.padding[1]

                start = np.array([0, pad_h, pad_w, 0], dtype='int32')

                pad_sizes = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
            else:
                pad_d = self.padding[0]
                pad_h = self.padding[1]
                pad_w = self.padding[2]

                start = np.array([0, pad_d, pad_h, pad_w, 0], dtype='int32')

                pad_sizes = ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0))

            size = np.array(ops[1].outputs[0].shape, dtype='int32')

            start_tensor = self.create_attr_tensor(start)
            size_tensor = self.create_attr_tensor(size)

            slice_out = ops[1].outputs[0]
            pad_array = np.pad(self.outputs[0].tensor, pad_sizes)
            slice_input = self.create_transform_tensor(pad_array, quantization=self.outputs[0].quantization)
            ops[1].outputs[0] = slice_input

            slice_op = tfl_ops.SliceOperator([slice_input, start_tensor, size_tensor], [slice_out])
            output_shape = slice_input.shape
            ops.insert(2, slice_op)

        # Output shape handling
        output_shape_tensor = self.create_attr_tensor(np.array(output_shape, dtype='int32'))
        conv_op.inputs.insert(0, output_shape_tensor)

        # Weight handling
        weight = conv_op.inputs[1]
        if weight_dim in (3, 4):
            nchw2chwn_perm = np.array([1, 2, 3, 0], dtype='int32')
        else:
            nchw2chwn_perm = np.array([2, 3, 4, 1, 0], dtype='int32')
        nchw2chwn_perm_tensor = self.create_attr_tensor(nchw2chwn_perm)
        reordered_weight = self.create_transform_tensor(
            np.transpose(weight.tensor, nchw2chwn_perm), quantization=weight.quantization
        )
        conv_op.inputs[1] = reordered_weight
        reorder_op = tfl_ops.TransposeOperator([weight, nchw2chwn_perm_tensor], [reordered_weight])
        ops.insert(1, reorder_op)

        # Bias handling
        if self.enable_mtk_ops:
            kernel_num = self.inputs[1].shape[0]

            if len(self.inputs) > 2 and self.inputs[2].shape[0] != kernel_num and self.inputs[2].shape[0] == 1:
                if conv_op.inputs[0].dtype == np.float32:
                    bias = torch.tensor([conv_op.inputs[2][0]] * kernel_num, dtype='float32')
                else:
                    bias = torch.tensor([conv_op.inputs[2][0]] * kernel_num, dtype='int32')

                conv_op.inputs.append(self.create_attr_tensor(bias))

            else:
                if len(self.inputs) == 2 or self.inputs[2] is None:
                    if conv_op.inputs[0].dtype == np.dtype('float32'):
                        bias = np.zeros((kernel_num,), dtype='float32')
                        q_args = None
                    else:
                        bias = np.zeros((kernel_num,), dtype='int32')
                else:
                    bias = self.inputs[2]

                q_args = None
                if bias.dtype != np.dtype('float32'):
                    per_tensor = weight_tensor.quantization.dim is None

                    # Bias handling
                    if per_tensor:
                        bias_scale = input_tensor.quantization.scale * weight_tensor.quantization.scale
                        bias_zero_point = 0
                        bias_dim = None
                    else:
                        bias_scale = [input_tensor.quantization.scale * s for s in weight_tensor.quantization.scale]
                        bias_zero_point = [0] * len(bias_scale)
                        bias_dim = 0

                    q_args = QuantizationParameters(bias_scale, bias_zero_point, bias_dim)

                conv_op.inputs.append(self.create_attr_tensor(bias, quantization=q_args))
        else:
            if len(self.inputs) > 2 and self.inputs[2] is not None:
                bias_tensor = self.inputs[2]
                add_out = ops[-2].outputs[0]
                bias_transform = self.create_transform_tensor(
                    add_out.tensor.copy(), quantization=self.outputs[0].quantization
                )
                ops[-2].outputs[0] = bias_transform
                ops.insert(len(ops) - 1, tfl_ops.AddOperator([bias_transform, bias_tensor], [add_out]))

        ops = prev_ops + ops + next_ops

        for op in ops:
            graph_converter.add_operator(op)

        graph_converter.try_restore_edges(mapping)
