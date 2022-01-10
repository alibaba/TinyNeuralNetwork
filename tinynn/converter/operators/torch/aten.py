import torch
import numpy as np

from .aten_schema import *
from .. import CommonGraph
from .. import tflite as tfl

import tflite as tfl_schema
import warnings

from tinynn.util.util import get_logger

log = get_logger(__name__, 'INFO')


class ATenLstmOperator(ATenLstmSchema):
    def lstm_input_helper(self,
                          input_tensors,
                          params_tensors,
                          has_biases,
                          param_start_index,
                          input_start_index,
                          layer_idx,
                          suffix):
        weight_ih_slices = torch.chunk(params_tensors[param_start_index], 4, 0)
        gates = ["input", "forget", "cell", "output"]
        for idx, (weight_ih, gate) in enumerate(zip(weight_ih_slices, gates)):
            input_tensors[input_start_index + idx] = self.create_attr_tensor(weight_ih)

        weight_hh_slices = torch.chunk(params_tensors[param_start_index + 1], 4, 0)
        for idx, (weight_hh, gate) in enumerate(zip(weight_hh_slices, gates)):
            input_tensors[input_start_index + 4 + idx] = self.create_attr_tensor(weight_hh)

        if has_biases:
            assert params_tensors[param_start_index + 2].dtype == torch.float32
            assert params_tensors[param_start_index + 3].dtype == torch.float32

            fused_bias = params_tensors[param_start_index + 2] + params_tensors[param_start_index + 3]
            fused_bias_slices = torch.chunk(fused_bias, 4, 0)
            for idx, (bias, gate) in enumerate(zip(fused_bias_slices, gates)):
                input_tensors[input_start_index + 11 + idx] = self.create_attr_tensor(bias)

    def lstm_hidden_state_helper(self,
                                 input_tensors,
                                 hidden_state_tensors,
                                 hidden_state_index,
                                 input_index,
                                 num_directions,
                                 direction_idx,
                                 num_layers,
                                 layer_idx,
                                 suffix,
                                 state_type):

        hidden_state_tensor = hidden_state_tensors[hidden_state_index]
        assert hidden_state_tensor.dim() == 3
        slice_idx = layer_idx * num_directions + direction_idx
        input_tensors[input_index] = self.create_attr_tensor(hidden_state_tensor[slice_idx])
        input_tensors[input_index].is_variable = True

    def parse_common(self,
                     input_tensor,
                     hidden_state_tensors,
                     params_tensors,
                     has_biases,
                     num_layers,
                     dropout,
                     is_train,
                     bidirectional,
                     batch_first,
                     graph_converter):
        assert is_train in (False, 0)
        expected_num_params = 2 * num_layers
        if has_biases:
            expected_num_params *= 2
        if bidirectional:
            expected_num_params *= 2

        assert len(
            params_tensors) == expected_num_params, f'num of params in LSTM is wrong. got: {len(params_tensors)}, expected: {expected_num_params}'

        num_input_tensors = 24
        num_directions = 1
        state_start_index = 18

        if bidirectional:
            num_input_tensors *= 2
            num_directions *= 2
            state_start_index = 35

        suffixes = ["_fw", "_bw"]
        state_kinds = ["act", "cell"]
        param_start_indices = [0, 4]
        input_start_indices = [1, 18]

        ops = []
        current_input = self.find_or_create_input(0, graph_converter)
        lstm_output = self.to_tfl_tensors(self.output_names[:1], self.output_tensors[:1])[0]
        params_offset = 0
        for layer_idx in range(num_layers):
            inputs = [current_input] + [tfl.OptionalTensorInstance] * (num_input_tensors - 1)

            for direction_idx in range(num_directions):
                self.lstm_input_helper(inputs,
                                       params_tensors,
                                       has_biases,
                                       params_offset + param_start_indices[direction_idx],
                                       input_start_indices[direction_idx],
                                       layer_idx,
                                       suffixes[direction_idx])

            for direction_idx in range(num_directions):
                for state_kind_idx in range(len(state_kinds)):
                    self.lstm_hidden_state_helper(inputs,
                                                  hidden_state_tensors,
                                                  state_kind_idx,
                                                  state_start_index + direction_idx * num_directions + state_kind_idx,
                                                  num_directions,
                                                  direction_idx,
                                                  num_layers,
                                                  layer_idx,
                                                  suffixes[direction_idx],
                                                  state_kinds[state_kind_idx])

            if layer_idx == num_layers - 1:
                layer_output = lstm_output
            else:
                output_shape = list(input_tensor.shape)
                output_shape[-1] = inputs[6].shape[1] * num_directions
                layer_output = self.create_transform_tensor(np.empty(output_shape, dtype=inputs[0].dtype))
            outputs = [layer_output]

            if bidirectional:
                ops.append(tfl.BidirectionalSequenceLstmOperator(inputs, outputs,
                                                                 fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                                                                 timeMajor=not batch_first,
                                                                 mergeOutputs=True))
            else:
                ops.append(tfl.UnidirectionalSequenceLstmOperator(inputs, outputs,
                                                                  fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                                                                  timeMajor=not batch_first))

            current_input = outputs[0]
            params_offset += 4 * num_directions

        for op in ops:
            graph_converter.add_operator(op)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor, hidden_state_tensors, params_tensors = self.input_tensors[:3]
        has_biases, num_layers, dropout, is_train, bidirectional, batch_first = self.input_tensors[3:]

        self.parse_common(input_tensor, hidden_state_tensors, params_tensors,
                          has_biases, num_layers, dropout, is_train, bidirectional,
                          batch_first, graph_converter)


class ATenBatchNormOperator(ATenBatchNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        eps = self.input_tensors[args['eps']]
        inputs = [self.find_or_create_input(i, graph_converter) for i in range(5)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.BatchNormOperator(inputs, outputs, eps))


class ATenConstantPadNdOperator(ATenConstantPadNdSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        pads = self.input_tensors[1]
        constant_value = self.input_tensors[2]

        orig_pad = np.array(pads, dtype='int32').reshape(-1, 2)
        pad_fill = np.zeros((input_tensor.tensor.ndim - orig_pad.shape[0], 2), dtype='int32')
        pad_arr = np.flip(np.concatenate((orig_pad, pad_fill)), 0)
        pad_tensor = self.create_attr_tensor(pad_arr)

        inputs = [input_tensor, pad_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        if constant_value not in (0, 0.0):
            output = outputs[0]
            if output.quantization is None:
                constant_arr = np.array([constant_value], dtype='float32')
            else:
                float_arr = torch.tensor([constant_value], dtype=torch.float32)
                constant_arr = torch.quantize_per_tensor(float_arr,
                                                         output.quantization.scale,
                                                         output.quantization.zero_point,
                                                         torch.quint8)

            inputs.append(self.create_attr_tensor(constant_arr))

            graph_converter.add_operator(tfl.Padv2Operator(inputs, outputs))
        else:
            graph_converter.add_operator(tfl.PadOperator(inputs, outputs))


class ATenUpsampleNearest2dOperator(ATenUpsampleNearest2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        output_size = self.input_tensors[1]
        if output_size is None:
            scale_factors = np.array(self.input_tensors[2], dtype='float64')
            input_sizes = np.array(input_tensor.shape[2:], dtype='float64')
            output_size = (input_sizes * scale_factors).astype('int32')
        output_sizes = self.create_attr_tensor(np.array(output_size, dtype='int32'))

        inputs = [input_tensor, output_sizes]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = [tfl.ResizeNearestNeighborOperator(inputs, outputs, halfPixelCenters=True)]
        ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)

        for op in ops:
            graph_converter.add_operator(op)


class ATenUpsampleBilinear2dOperator(ATenUpsampleBilinear2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        output_size = self.input_tensors[1]
        if output_size is None:
            scale_factors = np.array(self.input_tensors[3], dtype='float64')
            input_sizes = np.array(input_tensor.shape[2:], dtype='float64')
            output_size = (input_sizes * scale_factors).astype('int32')
        output_sizes = self.create_attr_tensor(np.array(output_size, dtype='int32'))
        align_corners = self.input_tensors[2] in (True, 1)
        half_pixel_centers = not align_corners

        inputs = [input_tensor, output_sizes]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = [tfl.ResizeBilinearOperator(inputs, outputs, align_corners, half_pixel_centers)]
        ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)

        for op in ops:
            graph_converter.add_operator(op)


class ATenAvgPool2dOperator(ATenAvgPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        kernel_h, kernel_w = self.input_tensors[1]
        stride_h, stride_w = self.input_tensors[2] or self.input_tensors[1]
        padding_h, padding_w = self.input_tensors[3]
        ceil_mode = self.input_tensors[4]
        count_include_pad = self.input_tensors[5]
        divisor_override = self.input_tensors[6]

        assert count_include_pad in (True, 1)
        assert divisor_override is None or divisor_override == kernel_h == kernel_w

        padding = tfl_schema.Padding.VALID

        avgpool_op = tfl.AveragePool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h)
        ops = self.wrap_ops_with_nhwc_nchw_transposes([avgpool_op])
        self.handle_padding(padding_h, padding_w, 1, ops, ceil_mode)

        for op in ops:
            graph_converter.add_operator(op)


class ATenAdaptiveAvgPool2dOperator(ATenAdaptiveAvgPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        output_h, output_w = self.input_tensors[1]

        dim_h, dim_w = input_tensor.shape[2:]
        assert dim_h % output_h == 0 and dim_w % output_w == 0, f'not supported: input dim: [{dim_h}, {dim_w}], output size: [{output_h}, {output_w}]'
        assert input_tensor.tensor.ndim == 4, 'Only 4D input is supported'

        ops = []

        dims = self.create_attr_tensor(np.array([1, 2], dtype='int32'))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if output_h == 1 and output_w == 1:
            inputs = [input_tensor, dims]
            ops.append(tfl.MeanOperator(inputs, outputs, True))
        else:
            inputs = [input_tensor]
            padding = tfl_schema.Padding.VALID

            stride_h, stride_w = dim_h // output_h, dim_w // output_w
            kernel_h, kernel_w = dim_h - (output_h - 1) * stride_h, dim_w - (output_w - 1) * stride_w

            ops.append(tfl.AveragePool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h))

        ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)

        for op in ops:
            graph_converter.add_operator(op)


class ATenLeakyReluOperator(ATenLeakyReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.LeakyReluOperator, graph_converter, self.input_tensors[1])


class ATenEluOperator(ATenEluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert all(x == 1.0 for x in self.input_tensors[1:])
        self.elementwise_unary(tfl.EluOperator, graph_converter)


class ATenReciprocalOperator(ATenReciprocalSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        old_inp = self.input_tensors[0].to(dtype=torch.float32)
        self.input_tensors.clear()

        self.input_tensors.append(torch.tensor([1], dtype=old_inp.dtype))
        self.input_tensors.append(old_inp)

        self.elementwise_binary(tfl.DivOperator, graph_converter)


class ATenRsqrtOperator(ATenRsqrtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.RsqrtOperator, graph_converter)


class ATenHardtanhOperator(ATenHardtanhSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        min_value, max_value = self.input_tensors[1:]
        if min_value == 0 and max_value == 6:
            self.elementwise_unary(tfl.Relu6Operator, graph_converter)
        else:
            ops = []
            input_tensor = self.find_or_create_input(0, graph_converter)
            inter_tensor = self.create_transform_tensor(
                np.where(input_tensor.tensor > min_value, input_tensor.tensor, min_value))
            min_value_tensor = self.create_attr_tensor(np.array([min_value], dtype=input_tensor.dtype))
            ops.append(tfl.MaximumOperator([input_tensor, min_value_tensor], [inter_tensor]))

            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            max_value_tensor = self.create_attr_tensor(np.array([max_value], dtype=input_tensor.dtype))
            ops.append(tfl.MinimumOperator([inter_tensor, max_value_tensor], outputs))

            for op in ops:
                graph_converter.add_operator(op)


class ATenSubOperator(ATenSubSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inp = self.input_tensors[0]
        other = self.input_tensors[1]
        out = self.output_tensors[0]
        alpha = self.input_tensors[-1]
        assert alpha == 1

        if out.dtype != inp.dtype:
            casted = inp.clone().to(dtype=out.dtype)
            inp_t = self.find_or_create_input(0, graph_converter)
            if inp_t.buffer is None:
                new_inp = self.create_transform_tensor(casted)
                graph_converter.add_operator(tfl.CastOperator(
                    [inp_t], [new_inp], tfl.torch_tflite_dtype_mappings[inp.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                self.input_names[0] = new_inp.name
            self.input_tensors[0] = casted

        if type(other) == torch.Tensor:
            if out.dtype != other.dtype:
                casted = other.clone().to(dtype=out.dtype)
                other_t = self.find_or_create_input(1, graph_converter)
                if other_t.buffer is None:
                    new_other = self.create_transform_tensor(casted)
                    graph_converter.add_operator(tfl.CastOperator(
                        [other_t], [new_other], tfl.torch_tflite_dtype_mappings[other.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                    self.input_names[1] = new_other.name
                self.input_tensors[1] = casted
            self.elementwise_binary(tfl.SubOperator, graph_converter)
        elif type(other) in (int, float, bool):
            self.input_tensors[1] = np.array([other], dtype=self.input_tensors[0].detach().numpy().dtype)
            self.elementwise_binary(tfl.SubOperator, graph_converter)
        else:
            assert False, "other should have type int, float, tensor in aten::sub(input, other)"


class ATenTransposeOperator(ATenTransposeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        dim_1, dim_2 = self.input_tensors[1:]
        input_tensor = self.find_or_create_input(0, graph_converter)

        perm = np.arange(input_tensor.tensor.ndim, dtype='int32')
        perm[dim_1], perm[dim_2] = perm[dim_2], perm[dim_1]

        perm_tensor = self.create_attr_tensor(perm)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.TransposeOperator([input_tensor, perm_tensor], outputs))


class ATenMulOperator(ATenMulSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inp = self.input_tensors[0]
        other = self.input_tensors[1]
        out = self.output_tensors[0]

        if out.dtype != inp.dtype:
            casted = inp.clone().to(dtype=out.dtype)
            inp_t = self.find_or_create_input(0, graph_converter)
            if inp_t.buffer is None:
                new_inp = self.create_transform_tensor(casted)
                graph_converter.add_operator(tfl.CastOperator(
                    [inp_t], [new_inp], tfl.torch_tflite_dtype_mappings[inp.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                self.input_names[0] = new_inp.name
            self.input_tensors[0] = casted

        if type(other) == torch.Tensor:
            if out.dtype != other.dtype:
                casted = other.clone().to(dtype=out.dtype)
                other_t = self.find_or_create_input(1, graph_converter)
                if other_t.buffer is None:
                    new_other = self.create_transform_tensor(casted)
                    graph_converter.add_operator(tfl.CastOperator(
                        [other_t], [new_other], tfl.torch_tflite_dtype_mappings[other.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                    self.input_names[1] = new_other.name
                self.input_tensors[1] = casted
            self.elementwise_binary(tfl.MulOperator, graph_converter)
        elif type(other) in (int, float):
            self.input_tensors[1] = np.array([other], dtype=self.input_tensors[0].detach().numpy().dtype)
            self.elementwise_binary(tfl.MulOperator, graph_converter)
        else:
            assert False, "other should have type int, float, tensor in aten::mul(input, other)"


class ATenDequantizeOperator(ATenDequantizeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.DequantizeOperator, graph_converter)


class ATenQuantizePerTensorOperator(ATenQuantizePerTensorSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.QuantizeOperator, graph_converter)


class ATenFakeQuantizePerTensorAffineOperator(ATenFakeQuantizePerTensorAffineSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.passthrough(graph_converter)


class ATenFakeQuantizePerChannelAffineOperator(ATenFakeQuantizePerChannelAffineSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.passthrough(graph_converter)


class ATenFlipOperator(ATenFlipSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        n_dim = self.input_tensors[0].dim()
        dims = [x + n_dim if x < 0 else x for x in self.input_tensors[1]]
        self.input_tensors[1] = np.array(dims, dtype='int32')

        self.elementwise_binary(tfl.ReverseV2Operator, graph_converter)


class ATenDivOperator(ATenDivSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inp = self.input_tensors[0]
        other = self.input_tensors[1]
        out = self.output_tensors[0]

        if out.dtype != inp.dtype:
            casted = inp.clone().to(dtype=out.dtype)
            inp_t = self.find_or_create_input(0, graph_converter)
            if inp_t.buffer is None:
                new_inp = self.create_transform_tensor(casted)
                graph_converter.add_operator(tfl.CastOperator(
                    [inp_t], [new_inp], tfl.torch_tflite_dtype_mappings[inp.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                self.input_names[0] = new_inp.name
            self.input_tensors[0] = casted

        if type(other) == torch.Tensor:
            if out.dtype != other.dtype:
                casted = other.clone().to(dtype=out.dtype)
                other_t = self.find_or_create_input(1, graph_converter)
                if other_t.buffer is None:
                    new_other = self.create_transform_tensor(casted)
                    graph_converter.add_operator(tfl.CastOperator(
                        [other_t], [new_other], tfl.torch_tflite_dtype_mappings[other.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                    self.input_names[1] = new_other.name
                self.input_tensors[1] = casted
            self.elementwise_binary(tfl.DivOperator, graph_converter)
        elif type(other) in (int, float):
            self.input_tensors[1] = np.array([other], dtype=self.input_tensors[0].detach().numpy().dtype)
            self.elementwise_binary(tfl.DivOperator, graph_converter)
        else:
            assert False, "other should have type int, float, tensor in aten::mul(input, other)"


class ATenMeanOperator(ATenMeanSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)

        if 'dim' in args and 'keepdim' in args:
            dims, keep_dim = self.input_tensors[1:3]
            if type(dims) not in (list, tuple):
                dims = [dims]
        else:
            dims = list(range(input_tensor.tensor.ndim))
            keep_dim = False
            self.output_tensors[0] = self.output_tensors[0].view(1)

        for idx, dim in enumerate(dims):
            if dim < 0:
                dims[idx] += input_tensor.tensor.ndim

        ops = []
        transpose = False

        # If it is a pooling 2d op, consider wrapping it with transposes
        if len(input_tensor.shape) == 4 and dims == [2, 3]:
            dims = [1, 2]
            transpose = True

        dim_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))

        inputs = [input_tensor, dim_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops.append(tfl.MeanOperator(inputs, outputs, keep_dim))

        if transpose:
            if keep_dim:
                ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)
            else:
                orig_input = ops[0].inputs[0]

                nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')
                nchw2nhwc_perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)

                new_input = self.create_transform_tensor(np.transpose(
                    orig_input.tensor, nchw2nhwc_perm), quantization=orig_input.quantization)

                nchw2nhwc_transpose = tfl.TransposeOperator([orig_input, nchw2nhwc_perm_tensor], [new_input])

                ops[0].inputs[0] = new_input
                ops.insert(0, nchw2nhwc_transpose)

        for op in ops:
            graph_converter.add_operator(op)


class ATenPowOperator(ATenPowSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert self.input_tensors[0].dtype in (torch.float32, torch.int32)
        if type(self.input_tensors[1]) == torch.tensor:
            if self.input_tensors[1] != self.input_tensors[0].dtype:
                self.input_tensors[1] = self.input_tensors[1].to(dtype=self.input_tensors[0].dtype).reshape(1)
        else:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)
        self.elementwise_binary(tfl.PowOperator, graph_converter)


class ATenMaxPool2dOperator(ATenMaxPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        kernel_h, kernel_w = self.input_tensors[1]
        stride_h, stride_w = self.input_tensors[2]
        pad_h, pad_w = self.input_tensors[3]
        dilation_h, dilation_w = self.input_tensors[4]
        ceil_mode = self.input_tensors[5]

        assert dilation_h == dilation_w == 1

        padding = tfl_schema.Padding.VALID

        maxpool_op = tfl.MaxPool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h)
        ops = self.wrap_ops_with_nhwc_nchw_transposes([maxpool_op])
        self.handle_padding(pad_h, pad_w, 1, ops, ceil_mode)

        for op in ops:
            graph_converter.add_operator(op)


class ATenMatmulOperator(ATenMatmulSchema):
    def parse_common(self, node, attrs, args, graph_converter):
        input_tensor, weight_tensor = [self.find_or_create_input(i, graph_converter) for i in range(2)]
        if input_tensor.tensor.ndim >= 2 and input_tensor.tensor.ndim <= 5:
            if weight_tensor.tensor.ndim == 2:
                bias_tensor = self.create_attr_tensor(np.zeros(weight_tensor.shape[1], dtype='float32'))

                perm = [1, 0]
                perm_tensor = self.create_attr_tensor(np.array(perm, dtype='int32'))
                weight_transformed = self.create_transform_tensor(np.transpose(weight_tensor.tensor, perm))
                graph_converter.add_operator(tfl.TransposeOperator([weight_tensor, perm_tensor], [weight_transformed]))

                inputs = [input_tensor, weight_transformed, bias_tensor]
                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                keep_dims = len(outputs[0].shape) > 2
                graph_converter.add_operator(tfl.FullyConnectedOperator(inputs, outputs, keepNumDims=keep_dims))
            elif weight_tensor.tensor.ndim >= 2 and weight_tensor.tensor.ndim <= 5:
                self.elementwise_binary(tfl.BatchMatmulOperator, graph_converter)
        else:
            self.unimplemented(node, attrs, args)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


class ATenFlattenOperator(ATenFlattenSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.reshape(graph_converter)


class ATenDropoutOperator(ATenDropoutSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.passthrough(graph_converter)


class ATenSoftmaxOperator(ATenSoftmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim = self.input_tensors[1]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        assert dim == len(self.input_tensors[0].shape) - 1, "only softmax with last dim is supported"

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.SoftmaxOperator(inputs, outputs, 1.0))


class ATenAtan2Operator(ATenAtan2Schema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_binary(tfl.Atan2Operator, graph_converter)


class ATenSqrtOperator(ATenSqrtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.SqrtOperator, graph_converter)


class ATenAddmmOperator(ATenAddmmSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        bias_tensor, input_tensor, weight_tensor = [self.find_or_create_input(i, graph_converter) for i in range(3)]
        assert len(weight_tensor.shape) == 2, "Weight of AddMM should be 2D"

        perm = [1, 0]
        perm_tensor = self.create_attr_tensor(np.array(perm, dtype='int32'))
        weight_transformed = self.create_transform_tensor(np.transpose(weight_tensor.tensor, perm))
        graph_converter.add_operator(tfl.TransposeOperator([weight_tensor, perm_tensor], [weight_transformed]))

        inputs = [input_tensor, weight_transformed, bias_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        keep_dims = len(outputs[0].shape) > 2
        graph_converter.add_operator(tfl.FullyConnectedOperator(inputs, outputs, keepNumDims=keep_dims))


class ATenStackOperator(ATenStackSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        dim = self.input_tensors[1]
        assert type(dim) == int

        if dim < 0:
            dim += self.input_tensors[0][0].ndim + 1

        names = graph_converter.get_list_expanded_names(self.input_names[0])
        orig_inputs = self.to_tfl_tensors(names, self.input_tensors[0],
                                          graph_converter=graph_converter,
                                          non_existent_as_buffer=True)
        inputs = [self.create_transform_tensor(np.expand_dims(orig_inputs[i].tensor, dim))
                  for i in range(len(orig_inputs))]
        attrs = [self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in inputs]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = [tfl.ReshapeOperator([orig, attr], [new], new.tensor.shape)
               for orig, new, attr in zip(orig_inputs, inputs, attrs)]
        ops.append(tfl.ConcatenationOperator(inputs, outputs, dim))

        for op in ops:
            graph_converter.add_operator(op)


class ATenCatOperator(ATenCatSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        dim = self.input_tensors[1]
        assert type(dim) == int

        if dim < 0:
            dim += self.input_tensors[0][0].ndim

        names = graph_converter.get_list_expanded_names(self.input_names[0])
        inputs = self.to_tfl_tensors(names, self.input_tensors[0],
                                     graph_converter=graph_converter,
                                     non_existent_as_buffer=True)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.ConcatenationOperator(inputs, outputs, dim))


class ATenPreluOperator(ATenPreluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_binary(tfl.PreluOperator, graph_converter)


class ATenToOperator(ATenToSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.CastOperator, graph_converter,
                               tfl.torch_tflite_dtype_mappings[self.input_tensors[0].dtype],
                               tfl.torch_tflite_dtype_mappings[self.output_tensors[0].dtype])


class ATenViewOperator(ATenViewSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.reshape(graph_converter)


class ATenSinOperator(ATenSinSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.SinOperator, graph_converter)


class ATenUnsqueezeOperator(ATenUnsqueezeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.reshape(graph_converter)


class ATenFloorOperator(ATenFloorSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.FloorOperator, graph_converter)


class ATenFloorDivideOperator(ATenFloorDivideSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_binary(tfl.FloorDivOperator, graph_converter)


class ATenCosOperator(ATenCosSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.CosOperator, graph_converter)


class ATenConv2dOperator(ATenConv2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input, weight, bias, stride, padding, dilation, groups = self.input_tensors[:7]
        if bias is None:
            end_index = 2
        else:
            end_index = 3

        output_padding = [0] * 2

        inputs = [self.find_or_create_input(i, graph_converter) for i in range(end_index)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.GenericConvOperator(
            inputs, outputs, stride, padding, dilation, output_padding, groups))


class ATenConvolutionOperator(ATenConvolutionSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input, weight, bias, stride, padding, dilation, transpose, output_padding, groups = self.input_tensors[:9]
        if bias is None:
            end_index = 2
        else:
            end_index = 3

        inputs = [self.find_or_create_input(i, graph_converter) for i in range(end_index)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if transpose == 0:
            graph_converter.add_operator(tfl.GenericConvOperator(
                inputs, outputs, stride, padding, dilation, output_padding, groups))
        else:
            graph_converter.add_operator(tfl.GenericTransposeConvOperator(
                inputs, outputs, stride, padding, dilation, output_padding, groups))


class ATenSliceOperator(ATenSliceSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        dim, start, end, step = self.input_tensors[1:]

        if start is None:
            start = 0

        if end is None:
            end = input_tensor.tensor.shape[dim]

        if start < 0:
            start += input_tensor.tensor.shape[dim]

        if end < 0:
            end += input_tensor.tensor.shape[dim]

        if dim < 0:
            dim += input_tensor.tensor.ndim

        if start > end:
            end = start

        if end >= input_tensor.tensor.shape[dim]:
            end = input_tensor.tensor.shape[dim]

        starts = np.zeros(input_tensor.tensor.ndim, dtype='int32')
        starts[dim] = start

        start_tensor = self.create_attr_tensor(starts)

        if step != 1:
            # if True:
            ends = np.array(input_tensor.tensor.shape, dtype='int32')
            ends[dim] = end

            strides = np.ones(input_tensor.tensor.ndim, dtype='int32')
            strides[dim] = step

            end_tensor = self.create_attr_tensor(ends)
            stride_tensor = self.create_attr_tensor(strides)

            inputs = [input_tensor, start_tensor, end_tensor, stride_tensor]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(tfl.StridedSliceOperator(inputs, outputs))
        else:
            sizes = np.array(input_tensor.tensor.shape, dtype='int32')
            sizes[dim] = end - start

            size_tensor = self.create_attr_tensor(sizes)
            inputs = [input_tensor, start_tensor, size_tensor]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(tfl.SliceOperator(inputs, outputs))


class ATenContiguousOperator(ATenContiguousSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.passthrough(graph_converter)


class ATenTOperator(ATenTSchema):
    def parse(self, node, attrs, args, graph_converter: CommonGraph):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.input_tensors[0]
        dims = len(input_tensor.shape)
        if dims >= 2:
            perm = torch.arange(dims).flip(dims=(0,))

            inputs = [self.find_or_create_input(0, graph_converter), self.create_attr_tensor(perm)]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(tfl.TransposeOperator(inputs, outputs))
        else:
            self.passthrough(graph_converter)


class ATenSqueezeOperator(ATenSqueezeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.reshape(graph_converter)


class ATenReshapeOperator(ATenReshapeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.reshape(graph_converter)


class ATenPermuteOperator(ATenPermuteSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        attr_tensor = self.create_attr_tensor(np.array(self.input_tensors[1], dtype='int32'))

        inputs = [input_tensor, attr_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.TransposeOperator(inputs, outputs))


class ATenAddOperator(ATenAddSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inp = self.input_tensors[0]
        other = self.input_tensors[1]
        alpha = self.input_tensors[-1]
        out = self.output_tensors[0]
        assert alpha == 1

        if out.dtype != inp.dtype:
            casted = inp.clone().to(dtype=out.dtype)
            inp_t = self.find_or_create_input(0, graph_converter)
            if inp_t.buffer is None:
                new_inp = self.create_transform_tensor(casted)
                graph_converter.add_operator(tfl.CastOperator(
                    [inp_t], [new_inp], tfl.torch_tflite_dtype_mappings[inp.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                self.input_names[0] = new_inp.name
            self.input_tensors[0] = casted

        if type(other) == torch.Tensor:
            if out.dtype != other.dtype:
                casted = other.clone().to(dtype=out.dtype)
                other_t = self.find_or_create_input(1, graph_converter)
                if other_t.buffer is None:
                    new_other = self.create_transform_tensor(casted)
                    graph_converter.add_operator(tfl.CastOperator(
                        [other_t], [new_other], tfl.torch_tflite_dtype_mappings[other.dtype], tfl.torch_tflite_dtype_mappings[out.dtype]))
                    self.input_names[1] = new_other.name
                self.input_tensors[1] = casted
            self.elementwise_binary(tfl.AddOperator, graph_converter)
        elif type(other) in (int, float, bool):
            self.input_tensors[1] = np.array([other], dtype=self.input_tensors[0].detach().numpy().dtype)
            self.elementwise_binary(tfl.AddOperator, graph_converter)
        else:
            assert False, "other should have type int, float, tensor in aten::add(input, other)"


class ATenReluOperator(ATenReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.ReluOperator, graph_converter)


class ATenRelu6Operator(ATenRelu6Schema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.Relu6Operator, graph_converter)


class ATenSigmoidOperator(ATenSigmoidSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.LogisticOperator, graph_converter)


class ATenSelectOperator(ATenSelectSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        dim, index = self.input_tensors[1:]

        assert type(dim) == int
        assert type(index) == int

        if dim < 0:
            dim += input_tensor.tensor.ndim

        if index < 0:
            index += input_tensor.tensor.shape[dim]

        index_tensor = self.create_attr_tensor(np.array([index], dtype='int32'))
        all_out = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]
        gather_out = self.create_transform_tensor(np.expand_dims(
            all_out.tensor, dim), quantization=all_out.quantization)
        reshape_attr = self.create_attr_tensor(self.output_tensors[0].shape)

        ops = []

        gather_inputs = [input_tensor, index_tensor]
        gather_outputs = [gather_out]
        ops.append(tfl.GatherOperator(gather_inputs, gather_outputs, dim))

        reshape_inputs = [gather_out, reshape_attr]
        reshape_outputs = [all_out]
        ops.append(tfl.ReshapeOperator(reshape_inputs, reshape_outputs, reshape_attr.tensor))

        for op in ops:
            graph_converter.add_operator(op)


class ATenTanhOperator(ATenTanhSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.TanhOperator, graph_converter)


class ATenEmbeddingOperator(ATenEmbeddingSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        weight, indices = [self.find_or_create_input(i, graph_converter) for i in range(2)]

        assert weight.tensor.ndim == 2
        assert indices.dtype in (np.int32, np.int64)

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.GatherOperator([weight, indices], outputs, 0))


class ATenLinearOperator(ATenLinearSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []
        input_tensor, weight_tensor, bias_tensor = self.input_tensors
        if input_tensor.dim() >= 2 and input_tensor.dim() <= 5 and bias_tensor is not None:
            # aten::addmm
            input_tensor, weight_tensor, bias_tensor = [self.find_or_create_input(i, graph_converter) for i in range(3)]
            assert len(weight_tensor.shape) == 2, "Weight of AddMM should be 2D"

            inputs = [input_tensor, weight_tensor, bias_tensor]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            keep_dims = len(outputs[0].shape) > 2
            ops.append(tfl.FullyConnectedOperator(inputs, outputs, keepNumDims=keep_dims))
        else:
            # aten::matmul + aten::add
            log.error(f'aten::linear is not supported for input shape {input_tensor.shape}, '
                      f'weight shape {weight_tensor.shape}, '
                      f'bias type {type(bias_tensor).__name__}')
            self.unimplemented(node, attrs, args)

        for op in ops:
            graph_converter.add_operator(op)


class ATenClampOperator(ATenClampSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        min_value, max_value = self.input_tensors[1:]
        assert min_value is not None
        assert max_value is not None
        if min_value == 0 and max_value == 6:
            self.elementwise_unary(tfl.Relu6Operator, graph_converter)
        else:
            ops = []
            input_tensor = self.find_or_create_input(0, graph_converter)
            inter_tensor = self.create_transform_tensor(
                np.where(input_tensor.tensor > min_value, input_tensor.tensor, min_value))
            min_value_tensor = self.create_attr_tensor(np.array([min_value], dtype=input_tensor.dtype))
            ops.append(tfl.MaximumOperator([input_tensor, min_value_tensor], [inter_tensor]))

            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            max_value_tensor = self.create_attr_tensor(np.array([max_value], dtype=input_tensor.dtype))
            ops.append(tfl.MinimumOperator([inter_tensor, max_value_tensor], outputs))

            for op in ops:
                graph_converter.add_operator(op)


class ATenExpOperator(ATenExpSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.ExpOperator, graph_converter)


class ATenLogOperator(ATenLogSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.LogOperator, graph_converter)


class ATenNeOperator(ATenNeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)
        self.elementwise_binary(tfl.NotEqualOperator, graph_converter)


class ATenSoftplusOperator(ATenSoftplusSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        beta = self.input_tensors[1]

        assert beta == 1.0, "Only beta=1.0 is supported for aten::softplus"
        warnings.warn('threshold is ignored when transforiming aten::softplus')

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        exp_out = self.create_transform_tensor(np.exp(input_tensor.tensor))
        ops.append(tfl.ExpOperator([input_tensor], [exp_out]))

        one_tensor = self.create_attr_tensor(np.ones((1,), dtype=exp_out.dtype))
        add_out = self.create_transform_tensor(exp_out.tensor + one_tensor.tensor)
        ops.append(tfl.AddOperator([exp_out, one_tensor], [add_out]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.LogOperator([add_out], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenLayerNormOperator(ATenLayerNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        normalized_shape = self.input_tensors[1]
        weight_tensor = self.find_or_create_input(2, graph_converter)
        bias_tensor = self.find_or_create_input(3, graph_converter)
        eps = self.input_tensors[4]

        ops = []

        axes = [input_tensor.tensor.ndim - i for i in range(len(normalized_shape), 0, -1)]
        dims_tensor = self.create_attr_tensor(np.array(axes, dtype='int32'))
        mean_tensor = self.create_transform_tensor(np.mean(input_tensor.tensor, axis=tuple(axes), keepdims=True))
        ops.append(tfl.MeanOperator([input_tensor, dims_tensor], [mean_tensor], keepDims=True))

        squared_diff = self.create_transform_tensor(np.power(input_tensor.tensor - mean_tensor.tensor, 2))
        ops.append(tfl.SquaredDifferenceOperator([input_tensor, mean_tensor], [squared_diff]))

        var_tensor = self.create_transform_tensor(np.mean(squared_diff.tensor, axis=tuple(axes), keepdims=True))
        ops.append(tfl.MeanOperator([squared_diff, dims_tensor], [var_tensor], keepDims=True))

        numerator = self.create_transform_tensor(input_tensor.tensor - mean_tensor.tensor)
        ops.append(tfl.SubOperator([input_tensor, mean_tensor], [numerator]))

        eps_tensor = self.create_attr_tensor(np.array([eps], dtype='float32'))
        with_eps = self.create_transform_tensor(var_tensor.tensor + eps_tensor.tensor)
        ops.append(tfl.AddOperator([var_tensor, eps_tensor], [with_eps]))

        denominator = self.create_transform_tensor(np.sqrt(with_eps.tensor))
        ops.append(tfl.SqrtOperator([with_eps], [denominator]))

        norm = self.create_transform_tensor(numerator.tensor / denominator.tensor)
        ops.append(tfl.DivOperator([numerator, denominator], [norm]))

        mul_out = self.create_transform_tensor(norm.tensor * weight_tensor.tensor)
        ops.append(tfl.MulOperator([norm, weight_tensor], [mul_out]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.AddOperator([mul_out, bias_tensor], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenInstanceNormOperator(ATenInstanceNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        inp = self.find_or_create_input(0, graph_converter)
        eps = self.input_tensors[args['eps']]

        weight, bias = self.input_tensors[1:3]
        affine = False
        track_running_stats = False
        if weight is not None and bias is not None:
            affine = True
            weight, bias = [self.find_or_create_input(i, graph_converter) for i in range(1, 3)]

        running_mean, running_var = self.input_tensors[3:5]
        if running_mean is not None and running_var is not None:
            track_running_stats = True
            running_mean, running_var = [self.find_or_create_input(i, graph_converter) for i in range(3, 5)]

        if affine and track_running_stats:
            inputs = [inp, weight, bias, running_mean, running_var]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            ops.append(tfl.BatchNormOperator(inputs, outputs, eps))
        else:
            assert track_running_stats is False, 'Instance norm with track_running_stats=True and affine=False is not supported'
            dims = len(inp.shape)
            axis = tuple(range(2, dims))
            axis_tensor = self.create_attr_tensor(np.array(axis, dtype='int32'))
            dim_ones = (1,) * (dims - 2)
            dims = self.create_attr_tensor(np.array(axis, dtype='int32'))
            mean = self.create_transform_tensor(np.mean(inp.tensor, axis=axis, keepdims=True))
            ops.append(tfl.MeanOperator([inp, axis_tensor], [mean], keepDims=True))

            squared_diff = self.create_transform_tensor(np.power(inp.tensor - mean.tensor, 2))
            ops.append(tfl.SquaredDifferenceOperator([inp, mean], [squared_diff]))

            var = self.create_transform_tensor(np.mean(squared_diff.tensor, axis=axis, keepdims=True))
            ops.append(tfl.MeanOperator([squared_diff, dims], [var], keepDims=True))

            numerator = self.create_transform_tensor(inp.tensor - mean.tensor)
            ops.append(tfl.SubOperator([inp, mean], [numerator]))

            eps_tensor = self.create_attr_tensor(np.array([eps], dtype='float32'))
            with_eps = self.create_transform_tensor(var.tensor + eps_tensor.tensor)
            ops.append(tfl.AddOperator([var, eps_tensor], [with_eps]))

            denominator = self.create_transform_tensor(np.sqrt(with_eps.tensor))
            ops.append(tfl.SqrtOperator([with_eps], [denominator]))

            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            if affine is False:
                ops.append(tfl.DivOperator([numerator, denominator], outputs))
            else:
                weight.tensor = weight.tensor.reshape(-1, *dim_ones)
                bias.tensor = bias.tensor.reshape(-1, *dim_ones)

                weight_tensor = self.create_attr_tensor(weight.tensor)
                bias_tensor = self.create_attr_tensor(bias.tensor)

                norm = self.create_transform_tensor(numerator.tensor / denominator.tensor)
                ops.append(tfl.DivOperator([numerator, denominator], [norm]))

                mul_out = self.create_transform_tensor(norm.tensor * weight_tensor.tensor)
                ops.append(tfl.MulOperator([norm, weight_tensor], [mul_out]))

                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                ops.append(tfl.AddOperator([mul_out, bias_tensor], outputs))

        for op in ops:
            print(op)
            graph_converter.add_operator(op)


class ATenIndexOperator(ATenIndexSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        indices = self.input_tensors[1]

        filtered_dims = [i for i, idx in enumerate(indices) if idx is not None]
        assert all((indices[i].dtype in (torch.int64, torch.int32) for i in filtered_dims))
        assert len(filtered_dims) == 1, "Multiple indices for aten::index is not supported"

        try:
            names = [graph_converter.get_list_expanded_names(self.input_names[1])]
        except KeyError:
            names = [self.get_unique_attr_name() for _ in indices]

        filtered_names = [names[i] for i in filtered_dims]
        filtered_tensors = [indices[i].to(dtype=torch.int32) for i in filtered_dims]

        input_tensor = self.find_or_create_input(0, graph_converter)
        indice_tensors = self.to_tfl_tensors(filtered_names, filtered_tensors,
                                             graph_converter=graph_converter,
                                             non_existent_as_buffer=True)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        actual_input = input_tensor
        actual_output = None
        for i, (dim, idx) in enumerate(zip(filtered_dims, indice_tensors)):
            if i == len(filtered_dims) - 1:
                actual_output = outputs[0]
            else:
                actual_output = self.create_transform_tensor(np.take(actual_input.tensor, idx.tensor, axis=dim))

            graph_converter.add_operator(tfl.GatherOperator([actual_input, idx], [actual_output], axis=dim))

            actual_input = actual_output


class ATenLogSoftmaxOperator(ATenLogSoftmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim = self.input_tensors[1]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        assert dim == len(self.input_tensors[0].shape) - 1, "only log_softmax with last dim is supported"
        self.elementwise_unary(tfl.LogSoftmaxOperator, graph_converter)


class ATenCloneOperator(ATenCloneSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.passthrough(graph_converter)


class ATenRepeatOperator(ATenRepeatSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []
        input_tensor = self.find_or_create_input(0, graph_converter)
        actual_input = input_tensor
        if input_tensor.buffer is None:
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            repeats = self.input_tensors[1]
            input_shape = input_tensor.shape
            if len(repeats) > len(input_shape):
                new_shape = [1] * (len(repeats) - len(input_shape)) + list(input_shape)
                new_shape_arr = np.array(new_shape, dtype='int32')
                reshaped = self.create_transform_tensor(np.reshape(input_tensor.tensor, new_shape_arr))
                actual_input = reshaped
                ops.append(tfl.ReshapeOperator([input_tensor], [reshaped], new_shape_arr))
            repeat_tensor = self.create_attr_tensor(np.array(repeats, dtype='int32'))
            ops.append(tfl.TileOperator([actual_input, repeat_tensor], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenMmOperator(ATenMmSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenMatmulOperator.parse_common(self, node, attrs, args, graph_converter)


class ATenHardswishOperator(ATenHardswishSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.HardSwishOperator, graph_converter)


class ATenHardsigmoidOperator(ATenHardsigmoidSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        three_tensor = self.create_attr_tensor(np.array([3], dtype=input_tensor.dtype))
        plus_three = self.create_transform_tensor(input_tensor.tensor + three_tensor.tensor)
        ops.append(tfl.AddOperator([input_tensor, three_tensor], [plus_three]))

        relu6_tensor = self.create_transform_tensor(np.clip(plus_three.tensor, 0, 6))
        ops.append(tfl.Relu6Operator([plus_three], [relu6_tensor]))

        six_tensor = self.create_attr_tensor(np.array([6], dtype=input_tensor.dtype))
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]
        ops.append(tfl.DivOperator([relu6_tensor, six_tensor], [output_tensor]))

        for op in ops:
            graph_converter.add_operator(op)


class ATenSiluOperator(ATenSiluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []
        input_tensor = self.find_or_create_input(0, graph_converter)
        sigmoid_x = self.create_transform_tensor(torch.sigmoid(torch.from_numpy(input_tensor.tensor)).numpy())
        ops.append(tfl.LogisticOperator([input_tensor], [sigmoid_x]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.MulOperator([input_tensor, sigmoid_x], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenVarOperator(ATenVarSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_dims = self.input_tensors[0].dim()

        dims = self.input_tensors[args['dim']] if 'dim' in args else list(range(input_dims))
        keep_dims = self.input_tensors[args['keepdim']] if 'keepdim' in args else False
        unbiased = self.input_tensors[args['unbiased']] if 'unbiased' in args else True
        correction = self.input_tensors[args['correction']] if 'correction' in args else 1

        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] += input_dims

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        ops = []

        sample_dims = [input_tensor.shape[i] for i in range(input_dims) if i in dims]
        samples = np.prod(sample_dims, dtype='float32')
        if unbiased and correction != 0:
            samples -= correction
        samples_tensor = self.create_attr_tensor(samples)

        dims_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))
        mean_tensor = self.create_transform_tensor(np.mean(input_tensor.tensor, axis=tuple(dims), keepdims=keep_dims))
        ops.append(tfl.MeanOperator([input_tensor, dims_tensor], [mean_tensor], keepDims=keep_dims))

        squared_diff = self.create_transform_tensor(np.power(input_tensor.tensor - mean_tensor.tensor, 2))
        ops.append(tfl.SquaredDifferenceOperator([input_tensor, mean_tensor], [squared_diff]))

        if unbiased and correction != 0:
            squared_diff_sum = self.create_transform_tensor(
                np.sum(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims))
            ops.append(tfl.SumOperator([squared_diff, dims_tensor], [squared_diff_sum], keepDims=keep_dims))
            ops.append(tfl.DivOperator([squared_diff_sum, samples_tensor], [output_tensor]))
        else:
            ops.append(tfl.MeanOperator([squared_diff, dims_tensor], [output_tensor], keepDims=keep_dims))

        for op in ops:
            graph_converter.add_operator(op)


class ATenStdOperator(ATenStdSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_dims = self.input_tensors[0].dim()

        dims = self.input_tensors[args['dim']] if 'dim' in args else list(range(input_dims))
        keep_dims = self.input_tensors[args['keepdim']] if 'keepdim' in args else False
        unbiased = self.input_tensors[args['unbiased']] if 'unbiased' in args else True
        correction = self.input_tensors[args['correction']] if 'correction' in args else 1

        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] += input_dims

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        ops = []

        sample_dims = [input_tensor.shape[i] for i in range(input_dims) if i in dims]
        samples = np.prod(sample_dims, dtype='float32')
        if unbiased and correction != 0:
            samples -= correction
        samples_tensor = self.create_attr_tensor(samples)

        dims_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))
        mean_tensor = self.create_transform_tensor(np.mean(input_tensor.tensor, axis=tuple(dims), keepdims=keep_dims))
        ops.append(tfl.MeanOperator([input_tensor, dims_tensor], [mean_tensor], keepDims=keep_dims))

        squared_diff = self.create_transform_tensor(np.power(input_tensor.tensor - mean_tensor.tensor, 2))
        ops.append(tfl.SquaredDifferenceOperator([input_tensor, mean_tensor], [squared_diff]))

        if unbiased and correction != 0:
            squared_diff_sum = self.create_transform_tensor(
                np.sum(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims))
            ops.append(tfl.SumOperator([squared_diff, dims_tensor], [squared_diff_sum], keepDims=keep_dims))

            var_tensor = self.create_transform_tensor(squared_diff_sum.tensor / samples_tensor.tensor)
            ops.append(tfl.DivOperator([squared_diff_sum, samples_tensor], [var_tensor]))
        else:
            var_tensor = self.create_transform_tensor(
                np.mean(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims))
            ops.append(tfl.MeanOperator([squared_diff, dims_tensor], [var_tensor], keepDims=keep_dims))

        ops.append(tfl.SqrtOperator([var_tensor], [output_tensor]))

        for op in ops:
            graph_converter.add_operator(op)


class ATenReflectionPad2dOperator(ATenReflectionPad2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        pads = self.input_tensors[1]
        tfl_pads = np.array([[0, 0], [0, 0], [pads[2], pads[3]], [pads[0], pads[1]]], dtype='int32')
        pad_tensor = self.create_attr_tensor(tfl_pads)

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.MirrorPadOperator(
            [input_tensor, pad_tensor], outputs, tfl_schema.MirrorPadMode.REFLECT))


class ATenReflectionPad1dOperator(ATenReflectionPad1dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        pads = self.input_tensors[1]
        tfl_pads = np.array([[0, 0], [0, 0], [pads[0], pads[1]]], dtype='int32')
        pad_tensor = self.create_attr_tensor(tfl_pads)

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.MirrorPadOperator(
            [input_tensor, pad_tensor], outputs, tfl_schema.MirrorPadMode.REFLECT))


class ATenSplitOperator(ATenSplitSchema):
    def parse_common(self, node, attrs, args, graph_converter):
        input_tensor = self.find_or_create_input(0, graph_converter)
        dim = self.input_tensors[2]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))
        size_splits = np.array([t.size(dim) for t in self.output_tensors[0]], dtype='int32')
        chunks = len(size_splits)
        split_tensor = self.create_attr_tensor(size_splits)

        output_names = [f'{self.output_names[0]}:{i}' for i in range(chunks)]
        graph_converter.add_iterable_pair(self.output_names, output_names, 'input')
        outputs = self.to_tfl_tensors(output_names, self.output_tensors[0])

        graph_converter.add_operator(tfl.SplitVOperator([input_tensor, split_tensor, dim_tensor], outputs, chunks))

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


class ATenSplitWithSizesOperator(ATenSplitWithSizesSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenSplitOperator.parse_common(self, node, attrs, args, graph_converter)


class ATenChunkOperator(ATenChunkSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        chunks, dim = self.input_tensors[1:]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        dim_size = self.input_tensors[0].size(dim)
        if chunks > dim_size:
            chunks = dim_size

        input_tensor = self.find_or_create_input(0, graph_converter)
        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))

        output_names = [f'{self.output_names[0]}:{i}' for i in range(len(self.output_tensors[0]))]
        graph_converter.add_iterable_pair(self.output_names, output_names, 'input')
        outputs = self.to_tfl_tensors(output_names, self.output_tensors[0])

        if dim_size % chunks != 0:
            size_splits = np.array([t.size(dim) for t in self.output_tensors[0]], dtype='int32')
            split_tensor = self.create_attr_tensor(size_splits)
            graph_converter.add_operator(tfl.SplitVOperator([input_tensor, split_tensor, dim_tensor], outputs, chunks))
        else:
            graph_converter.add_operator(tfl.SplitOperator([dim_tensor, input_tensor], outputs, chunks))


class ATenPixelShuffleOperator(ATenPixelShuffleSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        upscale_factor = self.input_tensors[1]
        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        # The implementation of tf.depth_to_space and torch.pixel_shuffle is not the same.
        # The former one splits the output channel with (block_size, block_size, new_channel_size),
        # while the latter one with (new_channel_size, block_size, block_size).
        # Since TFLite has no support for transposes for >5D tensors, we need to use `tf.gather`
        # to reorder the elements in the channel dimension.

        ops.append(tfl.DepthToSpaceOperator([input_tensor], outputs, upscale_factor))

        ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)

        c = input_tensor.shape[1]
        bs = upscale_factor
        perm = np.arange(c).reshape(c // (bs ** 2), bs, bs).transpose(1, 2, 0).flatten()
        if not np.array_equal(np.sort(perm), perm):
            reordered = self.create_transform_tensor(ops[0].outputs[0].tensor[:, :, :, perm])
            indices = self.create_attr_tensor(perm.astype('int32'))
            gather_op = tfl.GatherOperator([ops[0].inputs[0], indices], [reordered], axis=3)
            ops[0].inputs[0] = reordered
            ops.insert(0, gather_op)
        for op in ops:
            graph_converter.add_operator(op)


class ATenPixelUnshuffleOperator(ATenPixelUnshuffleSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        downscale_factor = self.input_tensors[1]
        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        # The implementation of tf.space_to_depth and torch.pixel_unshuffle is not the same.
        # The former one splits the output channel with (block_size, block_size, new_channel_size),
        # while the latter one with (new_channel_size, block_size, block_size).
        # Since TFLite has no support for transposes for >5D tensors, we need to use `tf.gather`
        # to reorder the elements in the channel dimension.
        ops.append(tfl.SpaceToDepthOperator([input_tensor], outputs, downscale_factor))

        ops = self.wrap_ops_with_nhwc_nchw_transposes(ops)

        c = input_tensor.shape[1]
        bs = downscale_factor
        perm = np.arange(c * (bs ** 2)).reshape(bs, bs, c).transpose(2, 0, 1).flatten()
        if not np.array_equal(np.sort(perm), perm):
            reordered = self.create_transform_tensor(ops[1].outputs[0].tensor[:, :, :, perm])
            indices = self.create_attr_tensor(perm.astype('int32'))
            gather_op = tfl.GatherOperator([reordered, indices], [ops[2].outputs[0]], axis=3)
            ops.append(gather_op)
            ops[2].outputs[0] = reordered

        for op in ops:
            graph_converter.add_operator(op)


class ATenArgmaxOperator(ATenArgmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim, keep_dim = self.input_tensors[1:3]

        # Downcast to int32
        self.output_tensors[0] = self.output_tensors[0].to(dtype=torch.int32)

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        if dim < 0:
            dim += input_tensor.tensor.ndim

        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))

        ops = []
        if keep_dim in (False, 0):
            ops.append(tfl.ArgMaxOperator([input_tensor, dim_tensor], [output_tensor], tfl_schema.TensorType.INT32))
        else:
            transform = self.create_transform_tensor(np.squeeze(output_tensor, dim))
            ops.append(tfl.ArgMaxOperator([input_tensor, dim_tensor], [transform], tfl_schema.TensorType.INT32))

            shape_tensor = self.create_attr_tensor(np.array(output_tensor.shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([transform, shape_tensor], [output_tensor], shape_tensor.tensor))

        for op in ops:
            graph_converter.add_operator(op)


class ATenArgminOperator(ATenArgminSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim, keep_dim = self.input_tensors[1:3]

        # Downcast to int32
        self.output_tensors[0] = self.output_tensors[0].to(dtype=torch.int32)

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        if dim < 0:
            dim += input_tensor.tensor.ndim

        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))

        ops = []
        if keep_dim in (False, 0):
            ops.append(tfl.ArgMinOperator([input_tensor, dim_tensor], [output_tensor], tfl_schema.TensorType.INT32))
        else:
            transform = self.create_transform_tensor(np.squeeze(output_tensor, dim))
            ops.append(tfl.ArgMinOperator([input_tensor, dim_tensor], [transform], tfl_schema.TensorType.INT32))

            shape_tensor = self.create_attr_tensor(np.array(output_tensor.shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([transform, shape_tensor], [output_tensor], shape_tensor.tensor))

        for op in ops:
            graph_converter.add_operator(op)


class ATenExpandOperator(ATenExpandSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        actual_input = input_tensor

        if input_tensor.buffer is None:
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            input_shape = input_tensor.shape
            output_shape = outputs[0].shape

            # No-OP if input tensor is already of desired sizes
            if output_shape == input_shape:
                return

            ops = []
            new_shape = input_shape
            actual_input = input_tensor
            if len(output_shape) > len(input_shape):
                new_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
                new_shape_arr = np.array(new_shape, dtype='int32')
                reshaped = self.create_transform_tensor(np.reshape(input_tensor.tensor, new_shape_arr))
                actual_input = reshaped
                ops.append(tfl.ReshapeOperator([input_tensor], [reshaped], new_shape_arr))

            repeats = []
            for x, y in zip(new_shape, output_shape):
                if x != y:
                    repeats.append(y)
                else:
                    repeats.append(1)

            repeat_tensor = self.create_attr_tensor(np.array(repeats, dtype='int32'))
            ops.append(tfl.TileOperator([actual_input, repeat_tensor], outputs))

            for op in ops:
                graph_converter.add_operator(op)


class ATenGatherOperator(ATenGatherSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        # torch.gather requires index tensor of type `torch.int64`
        orig_type = self.input_tensors[2].dtype
        self.input_tensors[2] = self.input_tensors[2].to(dtype=torch.int64)
        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]
        dim, index = self.input_tensors[1:3]
        if dim < 0:
            dim += input_tensor.tensor.ndim

        fake_input = torch.arange(input_tensor.tensor.size).reshape(input_tensor.shape)
        fake_output = torch.gather(fake_input, dim, index)

        indices = torch.nonzero(fake_input >= 0)[fake_output].to(dtype=torch.int32)

        self.input_tensors[2] = self.input_tensors[2].to(dtype=orig_type)
        index_tensor = self.find_or_create_input(2, graph_converter)
        if index_tensor.buffer is None:
            indices_per_dim = torch.split(indices, 1, dim=-1)
            indices_tensors = [self.create_attr_tensor(t) for t in indices_per_dim]

            index_shape = list(index_tensor.shape) + [1]
            axis = len(index_shape) - 1
            shape_tensor = self.create_attr_tensor(np.array(index_shape, dtype='int32'))
            index_reshaped = self.create_transform_tensor(np.reshape(index_tensor.tensor, index_shape))
            graph_converter.add_operator(tfl.ReshapeOperator([index_tensor, shape_tensor], [index_reshaped], index_shape))

            indices_tensors[dim] = index_reshaped
            indices_tensor = self.create_transform_tensor(np.concatenate([x.tensor for x in indices_tensors], axis=-1))
            graph_converter.add_operator(tfl.ConcatenationOperator(indices_tensors, [indices_tensor], axis=axis))
        else:
            indices_tensor = self.create_attr_tensor(indices)

        graph_converter.add_operator(tfl.GatherNdOperator([input_tensor, indices_tensor], [output_tensor]))


class ATenGeluOperator(ATenGeluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        constant_tensor = self.create_attr_tensor(np.array([1.702], dtype='float32'))
        sigmoid_in = self.create_transform_tensor(input_tensor.tensor * constant_tensor.tensor)
        ops.append(tfl.MulOperator([input_tensor, constant_tensor], [sigmoid_in]))

        sigmoid_out = self.create_transform_tensor(torch.sigmoid(torch.from_numpy(input_tensor.tensor)).numpy())
        ops.append(tfl.LogisticOperator([sigmoid_in], [sigmoid_out]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.MulOperator([sigmoid_out, input_tensor], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenCopyOperator(ATenCopySchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        ops = []
        if isinstance(other, torch.Tensor):
            other_tensor = self.find_or_create_input(1, graph_converter)

            if other_tensor.buffer is None:
                other_shape = other_tensor.shape
                output_shape = output_tensor.shape

                actual_input = other_tensor
                if other_tensor.dtype != output_tensor.dtype:
                    casted = self.create_transform_tensor(other_tensor.tensor.astype(output_tensor.dtype))
                    actual_input = casted
                    ops.append(tfl.CastOperator([other_tensor], [casted],
                                                inDataType=tfl.numpy_tflite_dtype_mappings[str(other_tensor.dtype)],
                                                outDataType=tfl.numpy_tflite_dtype_mappings[str(output_tensor.dtype)]))

                if other_shape == output_shape:
                    shape_tensor = self.create_attr_tensor(np.array(other_shape, dtype='int32'))
                    ops.append(tfl.ReshapeOperator([actual_input, shape_tensor], [output_tensor], shape_tensor.tensor))
                else:
                    new_shape = other_shape

                    if len(output_shape) > len(other_shape):
                        new_shape = [1] * (len(output_shape) - len(other_shape)) + list(other_shape)
                        new_shape_arr = np.array(new_shape, dtype='int32')
                        reshaped = self.create_transform_tensor(np.reshape(other_tensor.tensor, new_shape_arr))
                        ops.append(tfl.ReshapeOperator([actual_input], [reshaped], new_shape_arr))
                        actual_input = reshaped

                    repeats = []
                    for x, y in zip(new_shape, output_shape):
                        if x != y:
                            repeats.append(y)
                        else:
                            repeats.append(1)

                    repeat_tensor = self.create_attr_tensor(np.array(repeats, dtype='int32'))
                    ops.append(tfl.TileOperator([actual_input, repeat_tensor], [output_tensor]))

        for op in ops:
            graph_converter.add_operator(op)


class ATenQuantizedLstmOperator(ATenQuantizedLstmSchema, ATenLstmOperator):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor, hidden_state_tensors, params_tensors = self.input_tensors[:3]
        has_biases, num_layers, dropout, is_train, bidirectional, batch_first = self.input_tensors[3:9]

        params_l = []
        for t in params_tensors:
            weight_l = []
            bias_l = []
            params = self.unpack_params(t)[1][0]
            inner_params = params[-1]
            for p in inner_params:
                unpacked = self.unpack_params(p)[1]
                w = unpacked[0]
                weight_l.append(w[0])
                if len(w) > 1:
                    bias_l.append(w[1])
            params_l.extend(weight_l)
            params_l.extend(bias_l)

        self.parse_common(input_tensor, hidden_state_tensors, params_l,
                          has_biases, num_layers, dropout, is_train, bidirectional,
                          batch_first, graph_converter)


class ATenBmmOperator(ATenBmmSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenMatmulOperator.parse_common(self, node, attrs, args, graph_converter)


class ATenEqOperator(ATenEqSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)
        self.elementwise_binary(tfl.EqualOperator, graph_converter)


class ATenNegOperator(ATenNegSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.NegOperator, graph_converter)


class ATenBitwiseNotOperator(ATenBitwiseNotSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        assert self.input_tensors[0].dtype == torch.bool, "Only bools are supported in aten::bitwise_not"

        self.elementwise_unary(tfl.LogicalNotOperator, graph_converter)


class ATenBitwiseAndOperator(ATenBitwiseAndSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        if not isinstance(other, torch.Tensor):
            self.input_tensors[1] = torch.tensor([other]).repeat(self.input_tensors[0].shape)

        assert all((t.dtype == torch.bool for t in self.input_tensors)), "Only bools are supported in aten::bitwise_not"

        self.elementwise_unary(tfl.LogicalAndOperator, graph_converter)


class ATenBitwiseOrOperator(ATenBitwiseOrSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        if not isinstance(other, torch.Tensor):
            self.input_tensors[1] = torch.tensor([other]).repeat(self.input_tensors[0].shape)

        assert all((t.dtype == torch.bool for t in self.input_tensors)), "Only bools are supported in aten::bitwise_not"

        self.elementwise_unary(tfl.LogicalOrOperator, graph_converter)


class ATenSumOperator(ATenSumSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)

        if 'dim' in args and 'keepdim' in args:
            dims, keep_dim = self.input_tensors[1:3]
            if type(dims) not in (list, tuple):
                dims = [dims]
        else:
            dims = list(range(input_tensor.tensor.ndim))
            keep_dim = False
            self.output_tensors[0] = self.output_tensors[0].view(1)

        for idx, dim in enumerate(dims):
            if dim < 0:
                dims[idx] += input_tensor.tensor.ndim

        dim_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))

        inputs = [input_tensor, dim_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.SumOperator(inputs, outputs, keep_dim))
