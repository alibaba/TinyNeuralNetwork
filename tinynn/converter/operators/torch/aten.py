import warnings

import numpy as np
import torch

from tinynn.util.util import get_logger

from ...schemas.tflite import schema_generated as tfl_schema
from ...schemas.torch.aten_schema import *
from .. import CommonGraph
from .. import tflite as tfl

log = get_logger(__name__, 'INFO')


class ATenLstmOperator(ATenLstmSchema):
    def lstm_input_helper(
        self, input_tensors, params_tensors, has_biases, param_start_index, input_start_index, layer_idx, suffix
    ):
        hybrid = isinstance(self, ATenQuantizedLstmOperator)
        weight_ih_slices = torch.chunk(params_tensors[param_start_index], 4, 0)
        gates = ["input", "forget", "cell", "output"]
        for idx, (weight_ih, gate) in enumerate(zip(weight_ih_slices, gates)):
            input_tensors[input_start_index + idx] = self.create_attr_tensor(weight_ih, hybrid=hybrid)

        weight_hh_slices = torch.chunk(params_tensors[param_start_index + 1], 4, 0)
        for idx, (weight_hh, gate) in enumerate(zip(weight_hh_slices, gates)):
            input_tensors[input_start_index + 4 + idx] = self.create_attr_tensor(weight_hh, hybrid=hybrid)

        if has_biases:
            assert params_tensors[param_start_index + 2].dtype == torch.float32
            assert params_tensors[param_start_index + 3].dtype == torch.float32

            fused_bias = params_tensors[param_start_index + 2] + params_tensors[param_start_index + 3]
            fused_bias_slices = torch.chunk(fused_bias, 4, 0)
            for idx, (bias, gate) in enumerate(zip(fused_bias_slices, gates)):
                input_tensors[input_start_index + 11 + idx] = self.create_attr_tensor(bias)
        else:
            bias_shape = input_tensors[input_start_index + 3].shape[:1]
            for idx, gate in enumerate(gates):
                bias = torch.zeros(bias_shape, dtype=torch.float32)
                input_tensors[input_start_index + 11 + idx] = self.create_attr_tensor(bias)

    def lstm_hidden_state_helper(
        self,
        input_tensors,
        hidden_state_tensors,
        hidden_state_index,
        input_index,
        num_directions,
        direction_idx,
        num_layers,
        layer_idx,
        suffix,
        state_type,
    ):

        hidden_state_tensor = hidden_state_tensors[hidden_state_index]
        assert hidden_state_tensor.dim() == 3
        slice_idx = layer_idx * num_directions + direction_idx
        input_tensors[input_index] = self.create_attr_tensor(hidden_state_tensor[slice_idx])
        input_tensors[input_index].is_variable = True

    def parse_common(
        self,
        input_tensor,
        hidden_state_tensors,
        params_tensors,
        has_biases,
        num_layers,
        dropout,
        is_train,
        bidirectional,
        batch_first,
        graph_converter,
    ):
        assert is_train in (False, 0)
        expected_num_params = 2 * num_layers
        params_step = 2
        if has_biases:
            expected_num_params *= 2
            params_step *= 2
        if bidirectional:
            expected_num_params *= 2

        assert (
            len(params_tensors) == expected_num_params
        ), f'num of params in LSTM is wrong. got: {len(params_tensors)}, expected: {expected_num_params}'

        num_input_tensors = 24
        num_directions = 1
        state_start_index = 18

        if bidirectional:
            num_input_tensors *= 2
            num_directions *= 2
            state_start_index = 35

        suffixes = ["_fw", "_bw"]
        state_kinds = ["act", "cell"]
        param_start_indices = [0, params_step]
        input_start_indices = [1, 18]

        ops = []
        current_input = self.find_or_create_input(0, graph_converter)
        lstm_output = self.to_tfl_tensors(self.output_names[:1], self.output_tensors[:1])[0]
        params_offset = 0
        for layer_idx in range(num_layers):
            inputs = [current_input] + [tfl.OptionalTensorInstance] * (num_input_tensors - 1)

            for direction_idx in range(num_directions):
                self.lstm_input_helper(
                    inputs,
                    params_tensors,
                    has_biases,
                    params_offset + param_start_indices[direction_idx],
                    input_start_indices[direction_idx],
                    layer_idx,
                    suffixes[direction_idx],
                )

            for direction_idx in range(num_directions):
                for state_kind_idx in range(len(state_kinds)):
                    self.lstm_hidden_state_helper(
                        inputs,
                        hidden_state_tensors,
                        state_kind_idx,
                        state_start_index + direction_idx * num_directions + state_kind_idx,
                        num_directions,
                        direction_idx,
                        num_layers,
                        layer_idx,
                        suffixes[direction_idx],
                        state_kinds[state_kind_idx],
                    )

            if layer_idx == num_layers - 1:
                layer_output = lstm_output
            else:
                output_shape = list(input_tensor.shape)
                output_shape[-1] = inputs[6].shape[1] * num_directions
                layer_output = self.create_transform_tensor(np.empty(output_shape, dtype=inputs[0].dtype))
            outputs = [layer_output]

            if bidirectional:
                if not self.map_bilstm_to_lstm:
                    ops.append(
                        tfl.BidirectionalSequenceLstmOperator(
                            inputs,
                            outputs,
                            fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                            timeMajor=not batch_first,
                            mergeOutputs=True,
                            asymmetricQuantizeInputs=self.hybrid_asymmetric_inputs,
                        )
                    )
                else:
                    fw_i_end = input_start_indices[-1]
                    fw_s_start = state_start_index
                    fw_s_end = state_start_index + len(state_kinds)
                    fw_pad = num_input_tensors // 2 - fw_s_end
                    fw_lstm_inputs = (
                        inputs[:fw_i_end] + inputs[fw_s_start:fw_s_end] + [tfl.OptionalTensorInstance] * fw_pad
                    )
                    fw_out, bw_out = [
                        self.create_transform_tensor(t, quantization=outputs[0].quantization)
                        for t in np.split(outputs[0].tensor, 2, -1)
                    ]

                    ops.append(
                        tfl.UnidirectionalSequenceLstmOperator(
                            fw_lstm_inputs,
                            [fw_out],
                            fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                            timeMajor=not batch_first,
                            asymmetricQuantizeInputs=self.hybrid_asymmetric_inputs,
                        )
                    )

                    time_dim = 1 if batch_first else 0
                    bw_in = self.create_transform_tensor(np.flip(current_input.tensor, time_dim))
                    bw_dim = self.create_attr_tensor(np.array([time_dim], dtype='int32'))
                    ops.append(tfl.ReverseV2Operator([current_input, bw_dim], [bw_in]))

                    bw_raw_out = self.create_transform_tensor(np.flip(bw_out.tensor, time_dim))
                    bw_o_start = input_start_indices[-1]
                    bw_o_end = state_start_index
                    bw_s_start = state_start_index + len(state_kinds)
                    bw_s_end = state_start_index + len(state_kinds) * num_directions
                    bw_pad = num_input_tensors // 2 - bw_s_end
                    bw_lstm_inputs = (
                        [bw_in]
                        + inputs[bw_o_start:bw_o_end]
                        + inputs[bw_s_start:bw_s_end]
                        + [tfl.OptionalTensorInstance] * bw_pad
                    )

                    ops.append(
                        tfl.UnidirectionalSequenceLstmOperator(
                            bw_lstm_inputs,
                            [bw_raw_out],
                            fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                            timeMajor=not batch_first,
                            asymmetricQuantizeInputs=self.hybrid_asymmetric_inputs,
                        )
                    )

                    ops.append(tfl.ReverseV2Operator([bw_raw_out, bw_dim], [bw_out]))

                    ops.append(tfl.ConcatenationOperator([fw_out, bw_out], outputs, axis=2))
            else:
                ops.append(
                    tfl.UnidirectionalSequenceLstmOperator(
                        inputs,
                        outputs,
                        fusedActivationFunction=tfl_schema.ActivationFunctionType.TANH,
                        timeMajor=not batch_first,
                        asymmetricQuantizeInputs=self.hybrid_asymmetric_inputs,
                    )
                )

            current_input = outputs[0]
            params_offset += params_step * num_directions

        for op in ops:
            graph_converter.add_operator(op)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor, hidden_state_tensors, params_tensors = self.input_tensors[:3]
        has_biases, num_layers, dropout, is_train, bidirectional, batch_first = self.input_tensors[3:]

        self.parse_common(
            input_tensor,
            hidden_state_tensors,
            params_tensors,
            has_biases,
            num_layers,
            dropout,
            is_train,
            bidirectional,
            batch_first,
            graph_converter,
        )


class ATenBatchNormOperator(ATenBatchNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        eps = self.input_tensors[args['eps']]

        # weight
        if self.input_tensors[1] is None:
            self.input_names[1] = self.get_unique_attr_name()
            self.input_tensors[1] = torch.ones(self.input_tensors[0].size(1), dtype=torch.float32)

        # bias
        if self.input_tensors[2] is None:
            self.input_names[2] = self.get_unique_attr_name()
            self.input_tensors[2] = torch.zeros(self.input_tensors[0].size(1), dtype=torch.float32)

        # running mean & var
        assert (
            self.input_tensors[3] is not None and self.input_tensors[4] is not None
        ), "Running mean and variance should not be None. Please use LayerNorm instead."

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
                constant_arr = torch.quantize_per_tensor(
                    float_arr, output.quantization.scale, output.quantization.zero_point, torch.quint8
                )

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

        ops = [tfl.ResizeNearestNeighborOperator(inputs, outputs, halfPixelCenters=False)]
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
        stride_h, stride_w = self.input_tensors[2] or (kernel_h, kernel_w)
        padding_h, padding_w = self.input_tensors[3]
        ceil_mode = self.input_tensors[4] in (True, 1)
        count_include_pad = self.input_tensors[5] in (True, 1)
        divisor_override = self.input_tensors[6]

        assert divisor_override is None or divisor_override == kernel_h == kernel_w

        padding = tfl_schema.Padding.VALID

        avgpool_op = tfl.AveragePool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h)
        ops = self.wrap_ops_with_nhwc_nchw_transposes([avgpool_op])
        self.handle_padding(padding_h, padding_w, 1, ops, ceil_mode)

        if not count_include_pad:
            mask = 1.0 / torch.nn.functional.avg_pool2d(
                torch.ones_like(self.input_tensors[0]),
                (kernel_h, kernel_w),
                (stride_h, stride_w),
                (padding_h, padding_w),
                ceil_mode,
                count_include_pad=True,
            )
            mask_permuted = mask.permute(0, 2, 3, 1)
            mask_t = self.create_attr_tensor(mask_permuted)
            before_mask = outputs[0].tensor / mask_permuted
            before_mask_t = self.create_transform_tensor(before_mask)
            actual_out = ops[-2].outputs[0]
            ops[-2].outputs[0] = before_mask_t
            ops.insert(-1, tfl.MulOperator([before_mask_t, mask_t], [actual_out]))

        for op in ops:
            graph_converter.add_operator(op)


class ATenAdaptiveAvgPool2dOperator(ATenAdaptiveAvgPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        output_h, output_w = self.input_tensors[1]

        dim_h, dim_w = input_tensor.shape[2:]
        assert (
            dim_h % output_h == 0 and dim_w % output_w == 0
        ), f'not supported: input dim: [{dim_h}, {dim_w}], output size: [{output_h}, {output_w}]'
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


class ATenAdaptiveMaxPool2dOperator(ATenAdaptiveMaxPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        output_h, output_w = self.input_tensors[1]

        dim_h, dim_w = input_tensor.shape[2:]
        assert (
            dim_h % output_h == 0 and dim_w % output_w == 0
        ), f'not supported: input dim: [{dim_h}, {dim_w}], output size: [{output_h}, {output_w}]'
        assert input_tensor.tensor.ndim == 4, 'Only 4D input is supported'

        ops = []

        dims = self.create_attr_tensor(np.array([1, 2], dtype='int32'))

        log.warning(
            'OPs like`F.adaptive_maxpool_2d` have multiple outputs. However, only the first '
            'output will be preserved in our converter. If you need that tensor, please '
            'use the `torch.argmax` instead.'
        )

        outputs = self.to_tfl_tensors(self.output_names[:1], self.output_tensors[:1])

        if output_h == 1 and output_w == 1:
            inputs = [input_tensor, dims]
            ops.append(tfl.ReduceMaxOperator(inputs, outputs, True))
        else:
            inputs = [input_tensor]
            padding = tfl_schema.Padding.VALID

            stride_h, stride_w = dim_h // output_h, dim_w // output_w
            kernel_h, kernel_w = dim_h - (output_h - 1) * stride_h, dim_w - (output_w - 1) * stride_w

            ops.append(tfl.MaxPool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h))

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

        self.input_tensors.insert(0, torch.tensor([1], dtype=old_inp.dtype))
        self.input_names.insert(0, self.get_unique_attr_name())

        self.elementwise_binary(tfl.DivOperator, graph_converter, False)


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
                np.where(input_tensor.tensor > min_value, input_tensor.tensor, min_value)
            )
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

        other = self.input_tensors[1]
        alpha = self.input_tensors[-1]
        assert alpha == 1

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif type(other) != torch.Tensor:
            assert False, "other should have type int, float, tensor in aten::sub(input, other)"

        self.elementwise_binary(tfl.SubOperator, graph_converter, True)


class ATenRsubOperator(ATenRsubSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        alpha = self.input_tensors[-1]
        assert alpha == 1

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif type(other) != torch.Tensor:
            assert False, "other should have type int, float, tensor in aten::sub(input, other)"

        # Swap the first two input tensors and their names
        self.input_names[0], self.input_names[1] = self.input_names[1], self.input_names[0]
        self.input_tensors[0], self.input_tensors[1] = self.input_tensors[1], self.input_tensors[0]

        self.elementwise_binary(tfl.SubOperator, graph_converter, True)


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

        other = self.input_tensors[1]

        if type(other) in (int, float):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif type(other) != torch.Tensor:
            assert False, "other should have type int, float, tensor in aten::mul(input, other)"

        self.elementwise_binary(tfl.MulOperator, graph_converter, True)


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

        if len(dims) == 1:
            self.elementwise_binary(tfl.ReverseV2Operator, graph_converter, False)
        else:
            actual_input = self.find_or_create_input(0, graph_converter)
            for dim in dims[:-1]:
                transform = self.create_transform_tensor(np.flip(actual_input.tensor, dim))
                dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))
                graph_converter.add_operator(tfl.ReverseV2Operator([actual_input, dim_tensor], [transform]))
                actual_input = transform
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            dim_tensor = self.create_attr_tensor(np.array(dims[-1:], dtype='int32'))
            graph_converter.add_operator(tfl.ReverseV2Operator([actual_input, dim_tensor], outputs))


class ATenDivOperator(ATenDivSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        if type(other) in (int, float):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif type(other) != torch.Tensor:
            assert False, "other should have type int, float, tensor in aten::div(input, other)"

        self.elementwise_binary(tfl.DivOperator, graph_converter, True)


class ATenMeanOperator(ATenMeanSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.MeanOperator, args, graph_converter, True)


class ATenPowOperator(ATenPowSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert self.input_tensors[0].dtype in (torch.float32, torch.int32)

        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)

        self.elementwise_binary(tfl.PowOperator, graph_converter, True)


class ATenMaxPool2dOperator(ATenMaxPool2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        kernel_h, kernel_w = self.input_tensors[1]
        stride_h, stride_w = self.input_tensors[2] or (kernel_h, kernel_w)
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
                self.elementwise_binary(tfl.BatchMatmulOperator, graph_converter, False)
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

        train = self.input_tensors[args['train']]
        if train not in (0, False):
            log.warning('aten::dropout with train=True found and will add randomness to the model.')

            input_tensor = self.find_or_create_input(0, graph_converter)
            assert len(input_tensor.shape) in (2, 3), "Only supports dropout with 2d input for training mode"
            assert input_tensor.quantization is None, "Only supports dropout with floating input for training mode"

            p = self.input_tensors[args['p']]

            ops = []

            if len(input_tensor.shape) == 3:
                assert (
                    input_tensor.shape[0] == 1
                ), "Only supports dropout with 3d input with batch_size=1 for training mode"
            batch_size = input_tensor.shape[-2]
            num_samples = input_tensor.shape[-1]

            logits = self.create_attr_tensor(np.log(np.array([[p, 1 - p]] * batch_size, dtype='float32')))
            num_samples_tensor = self.create_attr_tensor(np.array(num_samples, dtype='int32'))
            multinomial_out = self.create_transform_tensor(np.empty((batch_size, num_samples), dtype='int32'))
            ops.append(tfl.MultinomialOperator([logits, num_samples_tensor], [multinomial_out]))

            casted = self.create_transform_tensor(np.empty((batch_size, num_samples), dtype='float32'))
            ops.append(
                tfl.CastOperator(
                    [multinomial_out],
                    [casted],
                    tfl.numpy_tflite_dtype_mappings[str(multinomial_out.dtype)],
                    tfl.numpy_tflite_dtype_mappings[str(casted.dtype)],
                )
            )

            scale = self.create_attr_tensor(np.array([1.0 / (1.0 - p)], dtype='float32'))
            scaled = self.create_transform_tensor(np.empty((batch_size, num_samples), dtype='float32'))
            ops.append(tfl.MulOperator([casted, scale], [scaled]))

            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            ops.append(tfl.MulOperator([input_tensor, scaled], outputs))

            for op in ops:
                graph_converter.add_operator(op)
        else:
            self.passthrough(graph_converter)


class ATenFeatureDropoutOperator(ATenFeatureDropoutSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        train = self.input_tensors[args['train']]
        if train not in (0, False):
            log.warning('aten::dropout with train=True found. Please check your model.')

        self.run(node)
        self.passthrough(graph_converter)


class ATenSoftmaxOperator(ATenSoftmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim = self.input_tensors[1]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        ops = []

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        softmax_op = tfl.SoftmaxOperator(inputs, outputs, 1.0)
        ops.append(softmax_op)

        ops = self.wrap_ops_with_last_dim_transposes(ops, dim)

        for op in ops:
            graph_converter.add_operator(op)


class ATenAtan2Operator(ATenAtan2Schema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_binary(tfl.Atan2Operator, graph_converter, False)


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
        orig_inputs = self.to_tfl_tensors(
            names, self.input_tensors[0], graph_converter=graph_converter, non_existent_as_buffer=True
        )
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        as_unpack = True
        for it in orig_inputs:
            if it.quantization is not None and outputs[0].quantization is not None:
                if (
                    it.quantization.scale != outputs[0].quantization.scale
                    or it.quantization.zero_point != outputs[0].quantization.zero_point
                    or it.quantization.dim != outputs[0].quantization.dim
                ):
                    as_unpack = False
                    break
            elif it.quantization is not None or outputs[0].quantization is not None:
                as_unpack = False
                break

        ops = []
        if as_unpack:
            ops.append(tfl.PackOperator(orig_inputs, outputs, len(orig_inputs), dim))
        else:
            inputs = [
                self.create_transform_tensor(np.expand_dims(orig_inputs[i].tensor, dim))
                for i in range(len(orig_inputs))
            ]
            attrs = [self.create_attr_tensor(np.array(t.shape, dtype='int32')) for t in inputs]

            ops.extend(
                [
                    tfl.ReshapeOperator([orig, attr], [new], new.tensor.shape)
                    for orig, new, attr in zip(orig_inputs, inputs, attrs)
                ]
            )

            for op in ops:
                op.extra_hints['direction'] = 'up'

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
        inputs = self.to_tfl_tensors(
            names, self.input_tensors[0], graph_converter=graph_converter, non_existent_as_buffer=True
        )
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.ConcatenationOperator(inputs, outputs, dim))


class ATenPreluOperator(ATenPreluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        alpha = self.input_tensors[1]

        weight_c = alpha.numel()
        input_c = self.input_tensors[0].shape[1]
        new_shape = [input_c] + [1] * (self.input_tensors[0].ndim - 2)

        alpha_tensor = self.find_or_create_input(1, graph_converter)
        shape_tensor = self.create_attr_tensor(np.array(new_shape, dtype='int32'))

        update_name = True
        if weight_c == input_c:
            new_alpha = self.create_transform_tensor(np.reshape(alpha_tensor.tensor, new_shape))
            graph_converter.add_operator(tfl.ReshapeOperator([alpha_tensor, shape_tensor], [new_alpha], new_shape))
        elif input_c != weight_c:
            new_alpha = self.create_transform_tensor(np.tile(alpha_tensor.tensor, new_shape))
            if alpha_tensor.buffer is None:
                graph_converter.add_operator(tfl.TileOperator([alpha_tensor, shape_tensor], [new_alpha]))
            else:
                update_name = False
                new_alpha = new_alpha.tensor

        self.input_tensors[1] = new_alpha
        if update_name:
            self.input_names[1] = new_alpha.name

        self.elementwise_binary(tfl.PreluOperator, graph_converter, False)


class ATenToOperator(ATenToSchema):
    def parse_common(self, node, attrs, args, graph_converter):
        out_type = self.output_tensors[0].dtype

        patch = False
        if out_type == torch.float64:
            patch = True
            out_type = torch.float32
            temp_tensor = self.output_tensors[0]
            self.output_tensors[0] = temp_tensor.detach().clone().to(dtype=torch.float32)

        self.elementwise_unary(
            tfl.CastOperator,
            graph_converter,
            tfl.torch_tflite_dtype_mappings[self.input_tensors[0].dtype],
            tfl.torch_tflite_dtype_mappings[out_type],
        )

        if patch:
            self.output_tensors[0] = temp_tensor

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


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
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)
        elif self.input_tensors[1].dtype != self.input_tensors[0].dtype:
            other = self.find_or_create_input(1, graph_converter)
            if other.buffer is None:
                new_other = self.input_tensors[1].detach().clone().to(dtype=self.input_tensors[0].dtype)
                new_other_t = self.create_transform_tensor(new_other)
                graph_converter.add_operator(
                    tfl.CastOperator(
                        [other],
                        [new_other_t],
                        tfl.torch_tflite_dtype_mappings[self.input_tensors[1].dtype],
                        tfl.torch_tflite_dtype_mappings[self.input_tensors[0].dtype],
                    )
                )
                self.input_tensors[1] = new_other
                self.input_names[1] = new_other_t.name
            else:
                self.input_tensors[1] = self.input_tensors[1].to(dtype=self.input_tensors[0].dtype)

        assert all(
            (not t.is_floating_point() for t in self.input_tensors[:2])
        ), "floor_divide for floats is not supported"

        assert all(
            ((t >= 0).all() for t in self.input_tensors[:2])
        ), "floor_divide for negative numbers is not supported"

        self.elementwise_binary(tfl.FloorDivOperator, graph_converter, False)


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

        graph_converter.add_operator(
            tfl.GenericConvOperator(inputs, outputs, stride, padding, dilation, output_padding, groups)
        )


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
            graph_converter.add_operator(
                tfl.GenericConvOperator(inputs, outputs, stride, padding, dilation, output_padding, groups)
            )
        else:
            graph_converter.add_operator(
                tfl.GenericTransposeConvOperator(
                    inputs, outputs, stride, padding, dilation, output_padding, groups, self.enable_mtk_ops
                )
            )


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

        other = self.input_tensors[1]
        alpha = self.input_tensors[-1]
        assert alpha == 1

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif type(other) != torch.Tensor:
            assert False, "other should have type int, float, tensor in aten::add(input, other)"

        self.elementwise_binary(tfl.AddOperator, graph_converter, True)


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
        gather_out = self.create_transform_tensor(
            np.expand_dims(all_out.tensor, dim), quantization=all_out.quantization
        )
        reshape_attr = self.create_attr_tensor(self.output_tensors[0].shape)

        ops = []

        gather_inputs = [input_tensor, index_tensor]
        gather_outputs = [gather_out]
        ops.append(tfl.GatherOperator(gather_inputs, gather_outputs, dim))

        reshape_inputs = [gather_out, reshape_attr]
        reshape_outputs = [all_out]
        reshape_op = tfl.ReshapeOperator(reshape_inputs, reshape_outputs, reshape_attr.tensor)
        reshape_op.extra_hints['direction'] = 'down'
        ops.append(reshape_op)

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
        if input_tensor.dim() >= 2 and input_tensor.dim() <= 5:
            assert len(weight_tensor.shape) == 2, "Weight of AddMM should be 2D"
            if bias_tensor is not None:
                input_tensor, weight_tensor, bias_tensor = [
                    self.find_or_create_input(i, graph_converter) for i in range(3)
                ]
            else:
                input_tensor, weight_tensor = [self.find_or_create_input(i, graph_converter) for i in range(2)]
                bias_tensor = self.create_attr_tensor(np.zeros(weight_tensor.shape[0], dtype='float32'))

            inputs = [input_tensor, weight_tensor, bias_tensor]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            keep_dims = len(outputs[0].shape) > 2
            ops.append(tfl.FullyConnectedOperator(inputs, outputs, keepNumDims=keep_dims))
        else:
            log.error(
                f'aten::linear is not supported for input shape {input_tensor.shape}, '
                f'weight shape {weight_tensor.shape}, '
                f'bias type {type(bias_tensor).__name__}'
            )
            self.unimplemented(node, attrs, args)

        for op in ops:
            graph_converter.add_operator(op)


class ATenClampOperator(ATenClampSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)

    def parse_common(self, node, attrs, args, graph_converter):
        if type(self) == ATenClampOperator:
            min_value, max_value = self.input_tensors[1:]
        elif type(self) == ATenClampMinOperator:
            min_value, max_value = self.input_tensors[1], None
        elif type(self) == ATenClampMaxOperator:
            min_value, max_value = None, self.input_tensors[1]

        has_min = min_value is not None
        has_max = max_value is not None

        assert has_min or has_max
        if min_value == 0 and max_value == 6:
            self.elementwise_unary(tfl.Relu6Operator, graph_converter)
        elif min_value == 0 and not has_max:
            self.elementwise_unary(tfl.ReluOperator, graph_converter)
        else:
            ops = []
            input_tensor = self.find_or_create_input(0, graph_converter)
            if has_min and has_max:
                inter_tensor = self.create_transform_tensor(
                    np.where(input_tensor.tensor > min_value, input_tensor.tensor, min_value)
                )
                min_value_tensor = self.create_attr_tensor(np.array([min_value], dtype=input_tensor.dtype))
                ops.append(tfl.MaximumOperator([input_tensor, min_value_tensor], [inter_tensor]))

                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                max_value_tensor = self.create_attr_tensor(np.array([max_value], dtype=input_tensor.dtype))
                ops.append(tfl.MinimumOperator([inter_tensor, max_value_tensor], outputs))
            elif has_min:
                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                max_value_tensor = self.create_attr_tensor(np.array([min_value], dtype=input_tensor.dtype))
                ops.append(tfl.MaximumOperator([input_tensor, max_value_tensor], outputs))
            else:
                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                max_value_tensor = self.create_attr_tensor(np.array([max_value], dtype=input_tensor.dtype))
                ops.append(tfl.MinimumOperator([input_tensor, max_value_tensor], outputs))

            for op in ops:
                graph_converter.add_operator(op)


class ATenClampMinOperator(ATenClampMinSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenClampOperator.parse_common(self, node, attrs, args, graph_converter)


class ATenClampMaxOperator(ATenClampMaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenClampOperator.parse_common(self, node, attrs, args, graph_converter)


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
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.NotEqualOperator, graph_converter, True)


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
            assert (
                track_running_stats is False
            ), 'Instance norm with track_running_stats=True and affine=False is not supported'
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
            names = graph_converter.get_list_expanded_names(self.input_names[1])
        except KeyError:
            names = [self.get_unique_attr_name() for _ in indices]

        filtered_names = [names[i] for i in filtered_dims]
        filtered_tensors = [indices[i].to(dtype=torch.int32) for i in filtered_dims]

        input_tensor = self.find_or_create_input(0, graph_converter)
        # TODO: support negative tensor indices
        filtered_tensors = [
            t + (t < 0).int() * input_tensor.shape[i] if n not in graph_converter.tensor_map else t
            for i, n, t in zip(filtered_dims, filtered_names, filtered_tensors)
        ]
        indice_tensors = self.to_tfl_tensors(
            filtered_names, filtered_tensors, graph_converter=graph_converter, non_existent_as_buffer=True
        )
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

        ops = []

        inputs = [self.find_or_create_input(0, graph_converter)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        log_softmax_op = tfl.LogSoftmaxOperator(inputs, outputs)
        ops.append(log_softmax_op)

        ops = self.wrap_ops_with_last_dim_transposes(ops, dim)

        for op in ops:
            graph_converter.add_operator(op)


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
                new_shape_tensor = self.create_attr_tensor(new_shape_arr)
                reshaped = self.create_transform_tensor(np.reshape(input_tensor.tensor, new_shape_arr))
                actual_input = reshaped
                ops.append(tfl.ReshapeOperator([input_tensor, new_shape_tensor], [reshaped], new_shape_arr))
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
        samples = samples.astype('float32')
        samples_tensor = self.create_attr_tensor(samples)

        dims_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))
        mean_tensor = self.create_transform_tensor(np.mean(input_tensor.tensor, axis=tuple(dims), keepdims=True))
        ops.append(tfl.MeanOperator([input_tensor, dims_tensor], [mean_tensor], keepDims=True))

        squared_diff = self.create_transform_tensor(np.power(input_tensor.tensor - mean_tensor.tensor, 2))
        ops.append(tfl.SquaredDifferenceOperator([input_tensor, mean_tensor], [squared_diff]))

        if unbiased and correction != 0:
            squared_diff_sum = self.create_transform_tensor(
                np.sum(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims)
            )
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
        samples = samples.astype('float32')
        samples_tensor = self.create_attr_tensor(samples)

        dims_tensor = self.create_attr_tensor(np.array(dims, dtype='int32'))
        mean_tensor = self.create_transform_tensor(np.mean(input_tensor.tensor, axis=tuple(dims), keepdims=True))
        ops.append(tfl.MeanOperator([input_tensor, dims_tensor], [mean_tensor], keepDims=True))

        squared_diff = self.create_transform_tensor(np.power(input_tensor.tensor - mean_tensor.tensor, 2))
        ops.append(tfl.SquaredDifferenceOperator([input_tensor, mean_tensor], [squared_diff]))

        if unbiased and correction != 0:
            squared_diff_sum = self.create_transform_tensor(
                np.sum(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims)
            )
            ops.append(tfl.SumOperator([squared_diff, dims_tensor], [squared_diff_sum], keepDims=keep_dims))

            var_tensor = self.create_transform_tensor(squared_diff_sum.tensor / samples_tensor.tensor)
            ops.append(tfl.DivOperator([squared_diff_sum, samples_tensor], [var_tensor]))
        else:
            var_tensor = self.create_transform_tensor(
                np.mean(squared_diff.tensor, axis=tuple(dims), keepdims=keep_dims)
            )
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
        graph_converter.add_operator(
            tfl.MirrorPadOperator([input_tensor, pad_tensor], outputs, tfl_schema.MirrorPadMode.REFLECT)
        )


class ATenReflectionPad1dOperator(ATenReflectionPad1dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        pads = self.input_tensors[1]
        tfl_pads = np.array([[0, 0], [0, 0], [pads[0], pads[1]]], dtype='int32')
        pad_tensor = self.create_attr_tensor(tfl_pads)

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(
            tfl.MirrorPadOperator([input_tensor, pad_tensor], outputs, tfl_schema.MirrorPadMode.REFLECT)
        )


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
            chunks = len(size_splits)
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
        perm = np.arange(c).reshape(c // (bs**2), bs, bs).transpose(1, 2, 0).flatten()
        if not np.array_equal(np.sort(perm), perm):
            reordered = self.create_transform_tensor(ops[0].outputs[0].tensor[:, :, :, perm])
            indices = self.create_attr_tensor(perm.astype('int32'))
            gather_op = tfl.GatherOperator([ops[0].outputs[0], indices], [reordered], axis=3)
            ops[1].inputs[0] = reordered
            ops.insert(1, gather_op)
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
        perm = np.arange(c * (bs**2)).reshape(bs, bs, c).transpose(2, 0, 1).flatten()
        if not np.array_equal(np.sort(perm), perm):
            reordered = self.create_transform_tensor(ops[1].outputs[0].tensor[:, :, :, perm])
            indices = self.create_attr_tensor(perm.astype('int32'))
            gather_op = tfl.GatherOperator([reordered, indices], [ops[1].outputs[0]], axis=3)
            ops.insert(2, gather_op)
            ops[1].outputs[0] = reordered

        for op in ops:
            graph_converter.add_operator(op)


class ATenArgmaxOperator(ATenArgmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        assert 'dim' in args and 'keepdim' in args, "aten::argmax(tensor) is not supported"

        # Downcast to int32
        self.output_tensors[0] = self.output_tensors[0].to(dtype=torch.int32)

        self.handle_reduce(tfl.ArgMaxOperator, args, graph_converter, False, tfl_schema.TensorType.INT32)


class ATenArgminOperator(ATenArgminSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        assert 'dim' in args and 'keepdim' in args, "aten::argmin(tensor) is not supported"

        # Downcast to int32
        self.output_tensors[0] = self.output_tensors[0].to(dtype=torch.int32)

        self.handle_reduce(tfl.ArgMinOperator, args, graph_converter, False, tfl_schema.TensorType.INT32)


class ATenExpandOperator(ATenExpandSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)

    def parse_common(self, node, attrs, args, graph_converter):
        input_tensor = self.find_or_create_input(0, graph_converter)
        actual_input = input_tensor

        if input_tensor.buffer is None:
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            input_shape = input_tensor.shape
            output_shape = outputs[0].shape

            # No-OP if input tensor is already of desired sizes
            if output_shape == input_shape:
                self.passthrough(graph_converter)
                return

            ops = []
            new_shape = input_shape
            actual_input = input_tensor
            if len(output_shape) > len(input_shape):
                new_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
                new_shape_arr = np.array(new_shape, dtype='int32')
                new_shape_tensor = self.create_attr_tensor(new_shape_arr)
                reshaped = self.create_transform_tensor(np.reshape(input_tensor.tensor, new_shape_arr))
                actual_input = reshaped
                reshape_op = tfl.ReshapeOperator([input_tensor, new_shape_tensor], [reshaped], new_shape_arr)
                reshape_op.extra_hints['direction'] = 'up'
                ops.append(reshape_op)

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


class ATenExpandAsOperator(ATenExpandAsSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenExpandOperator.parse_common(self, node, attrs, args, graph_converter)


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
            reshape_op = tfl.ReshapeOperator([index_tensor, shape_tensor], [index_reshaped], index_shape)
            reshape_op.extra_hints['direction'] = 'up'
            graph_converter.add_operator(reshape_op)

            if str(index_reshaped.dtype) != 'int32':
                index_casted = self.create_transform_tensor(index_reshaped.tensor.astype('int32'))
                graph_converter.add_operator(
                    tfl.CastOperator(
                        [index_reshaped],
                        [index_casted],
                        tfl.numpy_tflite_dtype_mappings[str(index_reshaped.dtype)],
                        tfl.numpy_tflite_dtype_mappings[str(index_casted.dtype)],
                    )
                )
                index_reshaped = index_casted

            indices_tensors[dim] = index_reshaped
            indices_tensor = self.create_transform_tensor(np.concatenate([x.tensor for x in indices_tensors], axis=-1))
            graph_converter.add_operator(tfl.ConcatenationOperator(indices_tensors, [indices_tensor], axis=axis))
        else:
            indices_tensor = self.create_attr_tensor(indices)

        graph_converter.add_operator(tfl.GatherNdOperator([input_tensor, indices_tensor], [output_tensor]))


class ATenScatterOperator(ATenScatterSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        assert not any(
            (torch.is_nonzero(v) for v in self.input_tensors[0].flatten())
        ), "aten::scatter with non-zero input is not supported"

        # torch.scatter requires index tensor of type `torch.int64`
        orig_type = self.input_tensors[2].dtype
        self.input_tensors[2] = self.input_tensors[2].to(dtype=torch.int64)
        self.run(node)

        assert 'reduce' not in args, "aten::scatter with reduction is not supported"

        input_tensor = self.find_or_create_input(0, graph_converter)
        assert input_tensor.buffer is not None, "aten::scatter with variable input is not supported"
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
            reshape_op = tfl.ReshapeOperator([index_tensor, shape_tensor], [index_reshaped], index_shape)
            reshape_op.extra_hints['direction'] = 'up'
            graph_converter.add_operator(reshape_op)

            if str(index_reshaped.dtype) != 'int32':
                index_casted = self.create_transform_tensor(index_reshaped.tensor.astype('int32'))
                graph_converter.add_operator(
                    tfl.CastOperator(
                        [index_reshaped],
                        [index_casted],
                        tfl.numpy_tflite_dtype_mappings[str(index_reshaped.dtype)],
                        tfl.numpy_tflite_dtype_mappings[str(index_casted.dtype)],
                    )
                )
                index_reshaped = index_casted

            indices_tensors[dim] = index_reshaped
            indices_tensor = self.create_transform_tensor(np.concatenate([x.tensor for x in indices_tensors], axis=-1))
            graph_converter.add_operator(tfl.ConcatenationOperator(indices_tensors, [indices_tensor], axis=axis))
        else:
            indices_tensor = self.create_attr_tensor(indices)

        if isinstance(self.input_tensors[3], (int, float)):
            fill_arr = np.squeeze(indices_tensor.tensor.copy().astype(input_tensor.dtype), -1)
            fill_arr.fill(self.input_tensors[3])
            fill_tensor = self.create_attr_tensor(fill_arr)
        else:
            val_tensor = self.find_or_create_input(3, graph_converter)

            val_slices = []
            for i in indices_tensor.shape:
                val_slices.append(slice(i))
            val_slices = tuple(val_slices[: len(val_tensor.shape)])

            val_sliced = val_tensor.tensor.__getitem__(val_slices)

            if val_tensor.buffer is None:
                if val_tensor.shape != indices_tensor.shape:
                    sizes = np.array(indices_tensor.tensor.shape[:-1], dtype='int32')
                    starts = np.zeros(indices_tensor.tensor.ndim - 1, dtype='int32')

                    size_tensor = self.create_attr_tensor(sizes)
                    start_tensor = self.create_attr_tensor(starts)

                    fill_tensor = self.create_transform_tensor(val_sliced)

                    graph_converter.add_operator(
                        tfl.SliceOperator([val_tensor, start_tensor, size_tensor], [fill_tensor])
                    )
            else:
                fill_tensor = self.create_attr_tensor(val_sliced)

        shape_tensor = self.create_attr_tensor(np.array(input_tensor.shape, dtype='int32'))

        graph_converter.add_operator(
            tfl.ScatterNdOperator([indices_tensor, fill_tensor, shape_tensor], [output_tensor])
        )


class ATenGeluOperator(ATenGeluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        constant_tensor = self.create_attr_tensor(np.array([1.702], dtype='float32'))
        sigmoid_in = self.create_transform_tensor(input_tensor.tensor * constant_tensor.tensor)

        actual_input = input_tensor
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]
        if input_tensor.quantization is not None:
            actual_input = self.create_transform_tensor(actual_input.tensor.astype('float32'))
            ops.append(tfl.DequantizeOperator([input_tensor], [actual_input]))

        ops.append(tfl.MulOperator([actual_input, constant_tensor], [sigmoid_in]))

        sigmoid_out = self.create_transform_tensor(torch.sigmoid(torch.from_numpy(input_tensor.tensor)).numpy())
        ops.append(tfl.LogisticOperator([sigmoid_in], [sigmoid_out]))

        if input_tensor.quantization is not None:
            actual_output = self.create_transform_tensor(output_tensor.tensor.astype('float32'))
            ops.append(tfl.MulOperator([sigmoid_out, actual_input], [actual_output]))
            ops.append(tfl.QuantizeOperator([actual_output], [output_tensor]))
        else:
            ops.append(tfl.MulOperator([sigmoid_out, actual_input], [output_tensor]))

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
                    ops.append(
                        tfl.CastOperator(
                            [other_tensor],
                            [casted],
                            inDataType=tfl.numpy_tflite_dtype_mappings[str(other_tensor.dtype)],
                            outDataType=tfl.numpy_tflite_dtype_mappings[str(output_tensor.dtype)],
                        )
                    )

                if other_shape == output_shape:
                    shape_tensor = self.create_attr_tensor(np.array(other_shape, dtype='int32'))
                    ops.append(tfl.ReshapeOperator([actual_input, shape_tensor], [output_tensor], shape_tensor.tensor))
                else:
                    new_shape = other_shape

                    if len(output_shape) > len(other_shape):
                        new_shape = [1] * (len(output_shape) - len(other_shape)) + list(other_shape)
                        new_shape_arr = np.array(new_shape, dtype='int32')
                        new_shape_tensor = self.create_attr_tensor(new_shape_arr)
                        reshaped = self.create_transform_tensor(np.reshape(actual_input.tensor, new_shape_arr))
                        reshape_op = tfl.ReshapeOperator([actual_input, new_shape_tensor], [reshaped], new_shape_arr)
                        reshape_op.extra_hints['direction'] = 'up'
                        ops.append(reshape_op)
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

        self.parse_common(
            input_tensor,
            hidden_state_tensors,
            params_l,
            has_biases,
            num_layers,
            dropout,
            is_train,
            bidirectional,
            batch_first,
            graph_converter,
        )


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
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.EqualOperator, graph_converter, True)


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
        self.parse_common(graph_converter)

    def parse_common(self, graph_converter):
        other = self.input_tensors[1]
        if not isinstance(other, torch.Tensor):
            self.input_tensors[1] = torch.tensor([other]).repeat(self.input_tensors[0].shape)

        assert all((t.dtype == torch.bool for t in self.input_tensors)), "Only bools are supported in aten::bitwise_not"

        self.elementwise_binary(tfl.LogicalAndOperator, graph_converter, False)


class ATenBitwiseOrOperator(ATenBitwiseOrSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(graph_converter)

    def parse_common(self, graph_converter):
        other = self.input_tensors[1]
        if not isinstance(other, torch.Tensor):
            self.input_tensors[1] = torch.tensor([other]).repeat(self.input_tensors[0].shape)

        assert all((t.dtype == torch.bool for t in self.input_tensors)), "Only bools are supported in aten::bitwise_not"

        self.elementwise_binary(tfl.LogicalOrOperator, graph_converter, False)


class ATenAndOperator(ATenAndSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenBitwiseAndOperator.parse_common(self, graph_converter)


class ATenOrOperator(ATenOrSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenBitwiseOrOperator.parse_common(self, graph_converter)


class ATenSumOperator(ATenSumSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.SumOperator, args, graph_converter, False)


class ATenProdOperator(ATenProdSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.ReduceProdOperator, args, graph_converter, False)


class ATenMinOperator(ATenMinSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.ReduceMinOperator, args, graph_converter, False)


class ATenMaxOperator(ATenMaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.ReduceMaxOperator, args, graph_converter, False)


class ATenAminOperator(ATenAminSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.ReduceMinOperator, args, graph_converter, False)


class ATenAmaxOperator(ATenAmaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.handle_reduce(tfl.ReduceMaxOperator, args, graph_converter, False)


class ATenGluOperator(ATenGluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        dim = self.input_tensors[1]

        if dim < 0:
            dim += input_tensor.tensor.ndim

        ops = []

        mid_arrs = np.split(input_tensor.tensor, 2, axis=dim)
        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))
        mid_tensors = [self.create_transform_tensor(t) for t in mid_arrs]
        ops.append(tfl.SplitOperator([dim_tensor, input_tensor], mid_tensors, 2))

        with_act = self.create_transform_tensor(torch.sigmoid(torch.from_numpy(mid_tensors[1].tensor)))
        ops.append(tfl.LogisticOperator([mid_tensors[1]], [with_act]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.MulOperator([mid_tensors[0], with_act], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenMaskedFillOperator(ATenMaskedFillSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        self.parse_common(graph_converter)

    def parse_common(self, graph_converter, input_idx=0, mask_idx=1, other_idx=2, out_idx=0):
        for i in (input_idx, other_idx):
            t = self.input_tensors[i]
            if type(t) == torch.Tensor:
                if t.dtype == torch.float64:
                    self.input_tensors[i] = t.to(dtype=torch.float32)
                elif t.dtype == torch.int64:
                    self.input_tensors[i] = t.to(dtype=torch.int32)

        if self.output_tensors[out_idx].dtype == torch.float64:
            self.output_tensors[out_idx] = self.output_tensors[out_idx].to(dtype=torch.float32)
        elif self.output_tensors[out_idx].dtype == torch.int64:
            self.output_tensors[out_idx] = self.output_tensors[out_idx].to(dtype=torch.int32)

        mask = self.input_tensors[mask_idx]
        other = self.input_tensors[other_idx]
        out = self.output_tensors[out_idx]

        input_tensor, mask_tensor = [self.find_or_create_input(i, graph_converter) for i in (input_idx, mask_idx)]

        ops = []
        if type(other) == torch.Tensor:
            other_t = self.find_or_create_input(other_idx, graph_converter)
            if out.dtype != other.dtype:
                casted = other.clone().to(dtype=out.dtype)
                if other_t.buffer is None:
                    new_other = self.create_transform_tensor(casted)
                    ops.append(
                        tfl.CastOperator(
                            [other_t],
                            [new_other],
                            tfl.torch_tflite_dtype_mappings[other.dtype],
                            tfl.torch_tflite_dtype_mappings[out.dtype],
                        )
                    )
                    other_t = new_other
                    # TODO: +/- inf check for variable tensors
                else:
                    if hasattr(torch.functional, 'atleast_1d'):
                        casted = torch.functional.atleast_1d(casted)
                    elif len(casted.shape) == 0:
                        casted = casted.reshape(1)
                    if torch.isinf(casted).any():
                        log.warning(
                            'aten::masked_fill(input, mask, value) where value=[+/-]inf is not supported, '
                            'trying to convert it to the nearest value'
                        )
                        type_info = torch.finfo(casted.dtype)
                        clamped = torch.clamp(casted, type_info.min, type_info.max)
                        other_t = self.create_attr_tensor(clamped, name=self.input_names[other_idx])
                    else:
                        other_t = self.create_attr_tensor(casted, name=self.input_names[other_idx])
        elif type(other) in (int, float):
            other_a = np.array([other], dtype=self.input_tensors[input_idx].detach().numpy().dtype)
            if np.isinf(other_a).any():
                log.warning(
                    'aten::masked_fill(input, mask, value) where value=[+/-]inf is not supported, '
                    'trying to convert it to the nearest value'
                )
                type_info = np.finfo(other_a.dtype)
                other_a = np.clip(other_a, type_info.min, type_info.max)
            other_t = self.create_attr_tensor(other_a)
        else:
            assert False, "value should have type float, tensor in aten::masked_fill(input, mask, value)"

        if mask_tensor.buffer is None:
            input_mask = self.create_transform_tensor(mask_tensor.tensor.astype(input_tensor.dtype))
            ops.append(
                tfl.CastOperator(
                    [mask_tensor],
                    [input_mask],
                    tfl.torch_tflite_dtype_mappings[mask.dtype],
                    tfl.torch_tflite_dtype_mappings[out.dtype],
                )
            )
        else:
            input_mask = self.create_attr_tensor(mask_tensor.tensor.astype(input_tensor.dtype))

        if mask_tensor.buffer is None or other_t.buffer is None:
            masked = self.create_transform_tensor(other_t.tensor * mask_tensor.tensor)
            ops.append(tfl.MulOperator([other_t, input_mask], [masked]))
        else:
            masked = self.create_attr_tensor(other_t.tensor * mask_tensor.tensor)

        one_tensor = self.create_attr_tensor(np.array([1], dtype=input_tensor.dtype))
        if mask_tensor.buffer is None:
            rev_mask = self.create_transform_tensor(one_tensor.tensor - mask_tensor.tensor)
            ops.append(tfl.SubOperator([one_tensor, input_mask], [rev_mask]))
        else:
            rev_mask = self.create_attr_tensor(one_tensor.tensor - mask_tensor.tensor)

        non_masked = self.create_transform_tensor(input_tensor.tensor * rev_mask.tensor)
        ops.append(tfl.MulOperator([input_tensor, rev_mask], [non_masked]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.AddOperator([non_masked, masked], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenGtOperator(ATenGtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.GreaterOperator, graph_converter, True)


class ATenLtOperator(ATenLtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.LessOperator, graph_converter, True)


class ATenGeOperator(ATenGeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.GreaterEqualOperator, graph_converter, np.True_)


class ATenLeOperator(ATenLeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.LessEqualOperator, graph_converter, True)


class ATenRemainderOperator(ATenRemainderSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if type(self.input_tensors[1]) != torch.Tensor:
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)

        self.elementwise_binary(tfl.FloorModOperator, graph_converter, True)


class ATenWhereOperator(ATenWhereSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert 'self' in args and 'other' in args, "aten::where(condition) is not supported"

        if type(self.input_tensors[2]) != torch.Tensor:
            self.input_tensors[2] = torch.tensor([self.input_tensors[2]])

        ATenMaskedFillOperator.parse_common(self, graph_converter, input_idx=2, mask_idx=0, other_idx=1)


class ATenTypeAsOperator(ATenTypeAsSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenToOperator.parse_common(self, node, attrs, args, graph_converter)


class ATenTopkOperator(ATenTopkSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor, k, dim, largest, sorted = self.input_tensors[:5]
        assert dim in (input_tensor.ndim - 1, -1), 'tflite topk only support last dim'
        assert largest in (1, True) and sorted in (1, True), 'tflite topk only support largest=True and sorted=True'
        input_tensor = self.find_or_create_input(0, graph_converter)
        k = self.create_attr_tensor(np.array([k], dtype='int32'))
        inputs = [input_tensor, k]
        self.output_tensors[1] = self.output_tensors[1].to(dtype=torch.int32)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        op = tfl.TopkV2Operator(inputs, outputs)
        graph_converter.add_operator(op)


class ATenCumsumOperator(ATenCumsumSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor, dim = self.input_tensors[:2]

        if dim < 0:
            dim += input_tensor.ndim

        input_tensor = self.find_or_create_input(0, graph_converter)
        dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))

        inputs = [input_tensor, dim_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        graph_converter.add_operator(tfl.CumsumOperator(inputs, outputs))


class ATenMeshgridOperator(ATenMeshgridSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert False, "aten::meshgrid for dynamic tensors is not supported"


class ATenFillOperator(ATenFillSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert False, "aten::fill_ for dynamic tensors is not supported"


class ATenUnbindOperator(ATenUnbindSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        dim = self.input_tensors[1]
        if dim < 0:
            dim += len(self.input_tensors[0].shape)

        chunks = self.input_tensors[0].shape[dim]
        output_names = [f'{self.output_names[0]}:{i}' for i in range(chunks)]
        graph_converter.add_iterable_pair(self.output_names, output_names, 'input')
        outputs = self.to_tfl_tensors(output_names, self.output_tensors[0])

        graph_converter.add_operator(tfl.UnpackOperator([input_tensor], outputs, chunks, dim))


class ATenRollOperator(ATenRollSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        shifts, dims = self.input_tensors[1:3]

        ops = []
        actual_input = input_tensor
        if len(dims) == 0:
            assert len(shifts) == 1
            shift = shifts[0]

            if len(input_tensor.shape) != 1:
                flat_input = self.create_transform_tensor(
                    input_tensor.tensor.ravel(), quantization=input_tensor.quantization
                )
                flat_shape = self.create_attr_tensor(np.array(flat_input.shape, dtype='int32'))

                prev_reshape_op = tfl.ReshapeOperator([input_tensor, flat_shape], [flat_input], flat_shape.tensor)
                prev_reshape_op.extra_hints['direction'] = 'up'
                ops.append(prev_reshape_op)

                actual_input = flat_input

            dims.append(0)

        assert len(shifts) == len(dims)
        for shift, dim in zip(shifts, dims):
            if dim < 0:
                dim += len(actual_input.shape)

            dim_size = actual_input.shape[dim]
            if shift < 0:
                shift += dim_size

            actual_shift = shift % dim_size
            if actual_shift != 0:
                split_sizes = self.create_attr_tensor(np.array([dim_size - actual_shift, actual_shift], dtype='int32'))
                dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))
                chunks = 2

                splitted = [
                    self.create_transform_tensor(x, quantization=actual_input.quantization)
                    for x in np.split(actual_input.tensor, [actual_shift], dim)
                ]
                ops.append(tfl.SplitVOperator([actual_input, split_sizes, dim_tensor], splitted, chunks))

                reversed_s = splitted[::-1]
                outputs = [
                    self.create_transform_tensor(
                        np.concatenate([s.tensor for s in reversed_s], dim), quantization=actual_input.quantization
                    )
                ]
                ops.append(tfl.ConcatenationOperator(reversed_s, outputs, dim))
            else:
                inputs = [actual_input, self.create_attr_tensor(actual_input.shape)]
                outputs = [
                    self.create_transform_tensor(actual_input.tensor.copy(), quantization=actual_input.quantization)
                ]
                ops.append(tfl.ReshapeOperator(inputs, outputs, input_tensor.shape))

            actual_input = outputs[0]

        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]
        if len(actual_input.shape) != len(output_tensor.shape):
            output_shape = self.create_attr_tensor(np.array(output_tensor.shape, dtype='int32'))
            post_reshape_op = tfl.ReshapeOperator([actual_input, output_shape], [output_tensor], output_shape.tensor)
            post_reshape_op.extra_hints['direction'] = 'down'
            ops.append(post_reshape_op)
        else:
            ops[-1].outputs[0] = output_tensor

        for op in ops:
            graph_converter.add_operator(op)


class ATenPadOperator(ATenPadSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        pads = self.input_tensors[1]
        mode = self.input_tensors[2]
        constant_value = self.input_tensors[3]

        op_cls_dict = {'constant': (tfl.PadOperator, tfl.Padv2Operator), 'reflect': (tfl.MirrorPadOperator, None)}
        assert mode in op_cls_dict, f"Unknown mode for aten::pad : {mode}"

        orig_pad = np.array(pads, dtype='int32').reshape(-1, 2)
        pad_fill = np.zeros((input_tensor.tensor.ndim - orig_pad.shape[0], 2), dtype='int32')
        pad_arr = np.flip(np.concatenate((orig_pad, pad_fill)), 0)
        pad_tensor = self.create_attr_tensor(pad_arr)

        inputs = [input_tensor, pad_tensor]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        if constant_value not in (0, 0.0, None):
            output = outputs[0]
            if output.quantization is None:
                constant_arr = np.array([constant_value], dtype='float32')
            else:
                float_arr = torch.tensor([constant_value], dtype=torch.float32)
                constant_arr = torch.quantize_per_tensor(
                    float_arr, output.quantization.scale, output.quantization.zero_point, torch.quint8
                )

            inputs.append(self.create_attr_tensor(constant_arr))

            graph_converter.add_operator(op_cls_dict[mode][1](inputs, outputs))
        else:
            graph_converter.add_operator(op_cls_dict[mode][0](inputs, outputs))


class ATenRoundOperator(ATenRoundSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.RoundOperator, graph_converter)


class ATenNormOperator(ATenNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        p = self.input_tensors[1]
        if p not in (1, 2):
            raise AssertionError("only torch.norm with p=1,2 is supported")

        input_t = self.find_or_create_input(0, graph_converter)

        if 'dim' in args and 'keepdim' in args:
            dims, keep_dim = self.input_tensors[2:4]
            if type(dims) not in (list, tuple):
                dims = [dims]
            if len(dims) == 0:
                dims = list(range(input_t.tensor.ndim))
                self.output_tensors[0] = self.output_tensors[0].view(1)
            elif len(dims) == input_t.tensor.ndim:
                self.output_tensors[0] = self.output_tensors[0].view(1)
        else:
            dims = list(range(input_t.tensor.ndim))
            keep_dim = False
            self.output_tensors[0] = self.output_tensors[0].view(1)

        for idx, dim in enumerate(dims):
            if dim < 0:
                dims[idx] += input_t.tensor.ndim

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        dim_t = self.create_attr_tensor(np.array(dims, dtype='int32'))

        ops = []
        if p == 1:
            tgt_t = self.create_transform_tensor(np.abs(input_t.tensor))
            ops.append(tfl.AbsOperator([input_t], [tgt_t]))
            actual_output = outputs[0]
        else:
            tgt_t = self.create_transform_tensor(np.power(input_t.tensor, 2))
            two_t = self.create_attr_tensor(np.array([2.0], dtype='float32'))
            ops.append(tfl.PowOperator([input_t, two_t], [tgt_t]))

            actual_output = self.create_transform_tensor(outputs[0].tensor)

        ops.append(tfl.SumOperator([tgt_t, dim_t], [actual_output], keepDims=keep_dim))

        if actual_output != outputs[0]:
            half_t = self.create_attr_tensor(np.array([0.5], dtype='float32'))
            ops.append(tfl.PowOperator([actual_output, half_t], outputs))

        for op in ops:
            graph_converter.add_operator(op)


class ATenAbsOperator(ATenAbsSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.AbsOperator, graph_converter)


class ATenIm2colOperator(ATenIm2colSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        assert input_tensor.tensor.ndim == 4, "only 4-D input tensors (batched image-like tensors) are supported"
        output_tensors = self.to_tfl_tensors(self.output_names, self.output_tensors)

        kernel_h, kernel_w = self.input_tensors[1]
        dilation_h, dilation_w = self.input_tensors[2]
        padding_h, padding_w = self.input_tensors[3]
        stride_h, stride_w = self.input_tensors[4]

        orig_pad = np.array([padding_h, padding_h, padding_w, padding_w], dtype='int32').reshape(-1, 2)
        pad_fill = np.zeros((input_tensor.tensor.ndim - orig_pad.shape[0], 2), dtype='int32')
        pad_arr = np.flip(np.concatenate((orig_pad, pad_fill)), 0)
        pad_tensor = self.create_attr_tensor(pad_arr)
        inter_tensor = self.create_transform_tensor(
            np.pad(input_tensor.tensor, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)))
        )
        graph_converter.add_operator(tfl.PadOperator([input_tensor, pad_tensor], [inter_tensor]))

        fake_input = torch.arange(0.0, inter_tensor.tensor.size).reshape(inter_tensor.shape)
        fake_output = torch.nn.functional.unfold(
            fake_input, (kernel_h, kernel_w), (dilation_h, dilation_w), (0, 0), (stride_h, stride_w)
        ).to(dtype=torch.int64)
        indices = torch.nonzero(fake_input >= 0)[fake_output].to(dtype=torch.int32)
        indices_tensor = self.create_attr_tensor(indices)
        graph_converter.add_operator(tfl.GatherNdOperator([inter_tensor, indices_tensor], output_tensors))


class ATenCol2imOperator(ATenCol2imSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        assert (
            input_tensor.tensor.ndim in (2, 3)
        ), "Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported"
        output_tensors = self.to_tfl_tensors(self.output_names, self.output_tensors)

        output_size_h, output_size_w = self.input_tensors[1]
        kernel_h, kernel_w = self.input_tensors[2]
        dilation_h, dilation_w = self.input_tensors[3]
        padding_h, padding_w = self.input_tensors[4]
        stride_h, stride_w = self.input_tensors[5]

        fold_out = torch.nn.functional.fold(
            torch.from_numpy(input_tensor.tensor), (output_size_h, output_size_w), (kernel_h, kernel_w),
            (dilation_h, dilation_w), (padding_h, padding_w), (stride_h, stride_w)
        )
        padded_fold_out = torch.nn.functional.pad(fold_out, (padding_w, padding_w, padding_h, padding_h)).numpy()
        fake_input = torch.arange(0.0, padded_fold_out.size).reshape(padded_fold_out.shape)
        if input_tensor.tensor.ndim == 2:
            fake_input = fake_input.unsqueeze(0)
        fake_output = torch.nn.functional.unfold(
            fake_input, (kernel_h, kernel_w), (dilation_h, dilation_w), (0, 0), (stride_h, stride_w)
        ).to(dtype=torch.int64)
        if input_tensor.tensor.ndim == 2:
            fake_input = fake_input.squeeze(0)
            fake_output = fake_output.squeeze(0)
        indices = torch.nonzero(fake_input >= 0)[fake_output].to(dtype=torch.int32)
        indices_tensor = self.create_attr_tensor(indices)
        shape_tensor = self.create_attr_tensor(np.array(padded_fold_out.shape, dtype='int32'))
        padded_fold_out_tensor = self.create_transform_tensor(padded_fold_out)
        graph_converter.add_operator(
            tfl.ScatterNdOperator([indices_tensor, input_tensor, shape_tensor], [padded_fold_out_tensor])
        )

        fake_input = torch.arange(0.0, padded_fold_out.size).reshape(padded_fold_out.shape)
        fake_output = fake_input[..., padding_h:output_size_h + padding_h, padding_w:output_size_w + padding_w].to(
            dtype=torch.int64)
        indices = torch.nonzero(fake_input >= 0)[fake_output].to(dtype=torch.int32)
        indices_tensor = self.create_attr_tensor(indices)
        graph_converter.add_operator(
            tfl.GatherNdOperator([padded_fold_out_tensor, indices_tensor], output_tensors)
        )


class ATenMishOperator(ATenMishSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        exp_out = self.create_transform_tensor(np.exp(input_tensor.tensor))
        ops.append(tfl.ExpOperator([input_tensor], [exp_out]))

        one_tensor = self.create_attr_tensor(np.ones((1,), dtype=exp_out.dtype))
        add_out = self.create_transform_tensor(exp_out.tensor + one_tensor.tensor)
        ops.append(tfl.AddOperator([exp_out, one_tensor], [add_out]))

        softplus_out = self.create_transform_tensor(np.log(add_out.tensor))
        ops.append(tfl.LogOperator([add_out], [softplus_out]))

        tanh_out = self.create_transform_tensor(np.tanh(softplus_out.tensor))
        ops.append(tfl.TanhOperator([softplus_out], [tanh_out]))

        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        ops.append(tfl.MulOperator([input_tensor, tanh_out], outputs))

        for op in ops:
            graph_converter.add_operator(op)
