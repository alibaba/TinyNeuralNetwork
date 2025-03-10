import warnings

import numpy as np
import torch

from tinynn.util.util import get_logger

from ...schemas.tflite import schema_generated as tfl_schema
from ...schemas.torch.aten_schema import *
from .. import CommonGraph
from .. import tflite as tfl

log = get_logger(__name__, 'INFO')


class AtenSignOperator(ATenSignSchema):
    def parse(self, node, attrs, args, graph_converter):

        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.SignOperator, graph_converter)


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
        tf_state_tensors,
    ):

        hidden_state_tensor = hidden_state_tensors[hidden_state_index]
        tf_state_tensor = tf_state_tensors[hidden_state_index]
        assert hidden_state_tensor.dim() == 3
        slice_idx = layer_idx * num_directions + direction_idx
        if tf_state_tensor[slice_idx] is None:
            input_tensors[input_index] = self.create_attr_tensor(hidden_state_tensor[slice_idx])
            input_tensors[input_index].is_variable = True
        else:
            assert self.unroll_rnn, "Input state tensors are only supported when unroll_rnn=True is specified"
            input_tensors[input_index] = tf_state_tensor[slice_idx]

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

        names = graph_converter.get_list_expanded_names(self.input_names[1])
        tf_in_state_tensors = [graph_converter.tensor_map.get(n, None) for n in names]
        tf_state_tensors = []
        unpacked_tensors = {}
        for t in tf_in_state_tensors:
            if t is not None and self.unroll_rnn:
                tensors = [
                    self.create_transform_tensor(np.squeeze(x, 0))
                    for x in np.split(t.tensor, num_directions * num_layers, 0)
                ]
                tf_state_tensors.append(tensors)
                ops.append(tfl.UnpackOperator([t], tensors, len(tensors), 0))
            else:
                tf_state_tensors.append([None] * num_directions * num_layers)

        current_input = self.find_or_create_input(0, graph_converter)
        lstm_output = self.to_tfl_tensors(self.output_names[:1], self.output_tensors[:1])[0]
        params_offset = 0
        tf_out_state_tensors = [[], []]
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
                        tf_state_tensors,
                    )

            if layer_idx == num_layers - 1:
                layer_output = lstm_output
            else:
                output_shape = list(input_tensor.shape)
                output_shape[-1] = inputs[6].shape[1] * num_directions
                layer_output = self.create_transform_tensor(np.empty(output_shape, dtype=inputs[0].dtype))
            outputs = [layer_output]

            if self.unroll_rnn:
                ts_axis = 1 if batch_first else 0
                num_timestep = inputs[0].shape[ts_axis]
                if inputs[0].name in unpacked_tensors:
                    input_ts = unpacked_tensors[inputs[0].name]
                else:
                    input_ts = [
                        self.create_transform_tensor(np.squeeze(x, ts_axis))
                        for x in np.split(inputs[0].tensor, num_timestep, ts_axis)
                    ]
                    ops.append(tfl.UnpackOperator([inputs[0]], input_ts, num_timestep, ts_axis))
                strides = [1, -1]

                output_ts = []
                for direction_idx in range(num_directions):
                    input_start = input_start_indices[direction_idx]
                    if not self.separated_rnn_gate_calc:
                        w_i = self.create_attr_tensor(
                            np.concatenate([inputs[x].tensor for x in range(input_start, input_start + 4)], 0),
                            quantization=inputs[input_start].quantization,
                        )
                        w_r = self.create_attr_tensor(
                            np.concatenate([inputs[x].tensor for x in range(input_start + 4, input_start + 8)], 0),
                            quantization=inputs[input_start + 4].quantization,
                        )
                        b_i = self.create_attr_tensor(
                            np.concatenate([inputs[x].tensor for x in range(input_start + 11, input_start + 15)], 0)
                        )
                        b_r = self.create_attr_tensor(np.zeros_like(b_i.tensor))
                    else:
                        w_i_list = [inputs[x] for x in range(input_start, input_start + 4)]
                        w_r_list = [inputs[x] for x in range(input_start + 4, input_start + 8)]
                        b_i_list = [inputs[x] for x in range(input_start + 11, input_start + 15)]
                        b_r_list = [self.create_attr_tensor(np.zeros_like(b_i.tensor)) for b_i in b_i_list]

                    state_start = state_start_index + direction_idx * num_directions
                    h = inputs[state_start]
                    c = inputs[state_start + 1]

                    stride = strides[direction_idx]

                    # Skip some computations for the first timestep
                    compute_h = h.buffer is None or np.any(h.tensor)
                    compute_c = c.buffer is None or np.any(c.tensor)

                    stacked_hs = []
                    for i, t in enumerate(input_ts[::stride]):
                        if not self.separated_rnn_gate_calc:
                            input_mm = self.create_transform_tensor(
                                np.matmul(t.tensor, np.transpose(w_i.tensor, [1, 0])) + b_i.tensor
                            )
                            ops.append(tfl.FullyConnectedOperator([t, w_i, b_i], [input_mm]))
                        else:
                            input_mm_list = []
                            for j, (w_i, b_i) in enumerate(zip(w_i_list, b_i_list)):
                                if j == 1 and i == 0 and not compute_c:
                                    input_mm_list.append(None)
                                    continue
                                input_mm = self.create_transform_tensor(
                                    np.matmul(t.tensor, np.transpose(w_i.tensor, [1, 0])) + b_i.tensor
                                )
                                ops.append(tfl.FullyConnectedOperator([t, w_i, b_i], [input_mm]))
                                input_mm_list.append(input_mm)

                        if i != 0 or compute_h:
                            if not self.separated_rnn_gate_calc:
                                hidden_mm = self.create_transform_tensor(
                                    np.matmul(h.tensor, np.transpose(w_r.tensor, [1, 0])) + b_r.tensor
                                )
                                ops.append(tfl.FullyConnectedOperator([h, w_r, b_r], [hidden_mm]))

                                add_out = self.create_transform_tensor(input_mm.tensor + hidden_mm.tensor)
                                ops.append(tfl.AddOperator([input_mm, hidden_mm], [add_out]))
                            else:
                                hidden_mm_list = []
                                for j, (w_r, b_r) in enumerate(zip(w_r_list, b_r_list)):
                                    if j == 1 and i == 0 and not compute_c:
                                        hidden_mm_list.append(None)
                                        continue
                                    hidden_mm = self.create_transform_tensor(
                                        np.matmul(h.tensor, np.transpose(w_r.tensor, [1, 0])) + b_r.tensor
                                    )
                                    ops.append(tfl.FullyConnectedOperator([h, w_r, b_r], [hidden_mm]))
                                    hidden_mm_list.append(hidden_mm)

                                gate_outs = []
                                for input_mm, hidden_mm in zip(input_mm_list, hidden_mm_list):
                                    if input_mm is not None and hidden_mm is not None:
                                        add_out = self.create_transform_tensor(input_mm.tensor + hidden_mm.tensor)
                                        ops.append(tfl.AddOperator([input_mm, hidden_mm], [add_out]))
                                        gate_outs.append(add_out)
                        else:
                            if not self.separated_rnn_gate_calc:
                                add_out = input_mm
                            else:
                                gate_outs = input_mm_list

                        if not self.separated_rnn_gate_calc:
                            gate_outs = [self.create_transform_tensor(t) for t in np.split(add_out.tensor, 4, 1)]
                            split_dim_tensor = self.create_attr_tensor(np.array(1, dtype='int32'))
                            ops.append(tfl.SplitOperator([split_dim_tensor, add_out], gate_outs, 4))

                        gate_i = self.create_transform_tensor(
                            torch.sigmoid(torch.from_numpy(gate_outs[0].tensor)).numpy()
                        )
                        ops.append(tfl.LogisticOperator([gate_outs[0]], [gate_i]))

                        if i != 0 or compute_c:
                            gate_f = self.create_transform_tensor(
                                torch.sigmoid(torch.from_numpy(gate_outs[1].tensor)).numpy()
                            )
                            ops.append(tfl.LogisticOperator([gate_outs[1]], [gate_f]))

                        gate_g = self.create_transform_tensor(np.tanh(gate_outs[2].tensor))
                        ops.append(tfl.TanhOperator([gate_outs[2]], [gate_g]))

                        gate_o = self.create_transform_tensor(
                            torch.sigmoid(torch.from_numpy(gate_outs[3].tensor)).numpy()
                        )
                        ops.append(tfl.LogisticOperator([gate_outs[3]], [gate_o]))

                        if i != 0 or compute_c:
                            c_left = self.create_transform_tensor(gate_f.tensor * c.tensor)
                            ops.append(tfl.MulOperator([gate_f, c], [c_left]))

                        c_right = self.create_transform_tensor(gate_i.tensor * gate_g.tensor)
                        ops.append(tfl.MulOperator([gate_i, gate_g], [c_right]))

                        if i != 0 or compute_c:
                            c = self.create_transform_tensor(c_left.tensor + c_right.tensor)
                            ops.append(tfl.AddOperator([c_left, c_right], [c]))
                        else:
                            c = c_right

                        c_act = self.create_transform_tensor(np.tanh(c.tensor))
                        ops.append(tfl.TanhOperator([c], [c_act]))

                        h = self.create_transform_tensor(gate_o.tensor * c_act.tensor)
                        ops.append(tfl.MulOperator([gate_o, c_act], [h]))

                        stacked_hs.append(h)

                    tf_out_state_tensors[0].append(h)
                    tf_out_state_tensors[1].append(c)

                    output_ts.extend(stacked_hs[::stride])

                if bidirectional:
                    # For bidirectional LSTMs, the forward output tensors and the backward output tensors are
                    # concatenated before we pack them together
                    fw_out = self.create_transform_tensor(
                        np.stack([x.tensor for x in output_ts[:num_timestep]], ts_axis)
                    )
                    ops.append(tfl.PackOperator(output_ts[:num_timestep], [fw_out], num_timestep, axis=ts_axis))

                    bw_out = self.create_transform_tensor(
                        np.stack([x.tensor for x in output_ts[num_timestep:]], ts_axis)
                    )
                    ops.append(tfl.PackOperator(output_ts[num_timestep:], [bw_out], num_timestep, axis=ts_axis))

                    ops.append(tfl.ConcatenationOperator([fw_out, bw_out], outputs, axis=2))
                elif layer_idx != num_layers - 1:
                    # Reusing unpacked tensors for the logic in the next layer
                    unpacked_tensors[outputs[0].name] = output_ts
                else:
                    # For the last layer, we have to pack the together
                    ops.append(tfl.PackOperator(output_ts, outputs, len(output_ts), axis=ts_axis))
            elif bidirectional:
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

        if self.unroll_rnn:
            state_outputs = self.to_tfl_tensors(self.output_names[1:], self.output_tensors[1:])
            for i, (orig, new) in enumerate(zip(tf_in_state_tensors, tf_out_state_tensors)):
                if orig is not None:
                    pack_op = tfl.PackOperator(new, state_outputs[i : i + 1], len(new), 0)
                    pack_op.extra_hints['warn_on_unused'] = False
                    ops.append(pack_op)
        else:
            ops[-1].extra_hints['cell_output'] = self.output_names[-1]
            common_names = set(self.output_names[1:]) & set(graph_converter.outputs)
            assert len(common_names) == 0, (
                f"Please remove the LSTM state outputs ({common_names}) from the model. Alternatively, you can try"
                " unroll_rnn=True"
            )

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


class ATenGruOperator(ATenGruSchema):
    def gru_input_helper(
        self, input_tensors, params_tensors, has_biases, param_start_index, input_start_index, layer_idx, suffix
    ):
        wir, wiz, win = torch.chunk(params_tensors[param_start_index], 3, 0)
        whr, whz, whn = torch.chunk(params_tensors[param_start_index + 1], 3, 0)

        wr = torch.cat((wir, whr), -1)
        wz = torch.cat((wiz, whz), -1)
        # [2*n_output, n_input+n_output]
        input_tensors[input_start_index] = self.create_attr_tensor(torch.cat((wr, wz), 0))
        # [n_output, n_input+n_output]
        input_tensors[input_start_index + 2] = self.create_attr_tensor(torch.cat((win, whn), -1))

        w_i_list = [self.create_attr_tensor(wir), self.create_attr_tensor(wiz), self.create_attr_tensor(win)]
        w_r_list = [self.create_attr_tensor(whr), self.create_attr_tensor(whz), self.create_attr_tensor(whn)]

        if has_biases:

            assert params_tensors[param_start_index + 2].dtype == torch.float32
            assert params_tensors[param_start_index + 3].dtype == torch.float32

            bir, biz, bin = torch.chunk(params_tensors[param_start_index + 2], 3, 0)
            bhr, bhz, bhn = torch.chunk(params_tensors[param_start_index + 3], 3, 0)

            br = torch.cat((bir, bhr), -1)
            bz = torch.cat((biz, bhz), -1)

            input_tensors[input_start_index + 1] = self.create_attr_tensor(torch.cat((br, bz), -1))  # [2*n_output]
            input_tensors[input_start_index + 3] = self.create_attr_tensor(torch.cat((bin, bhn), -1))  # [n_output]

            b_i_list = [self.create_attr_tensor(bir), self.create_attr_tensor(biz), self.create_attr_tensor(bin)]
            b_r_list = [self.create_attr_tensor(bhr), self.create_attr_tensor(bhz), self.create_attr_tensor(bhn)]

        else:

            bir = torch.zeros(input_tensors[input_start_index + 2].shape[0])
            biz = torch.zeros_like(bir)
            bin = torch.zeros_like(biz)
            bhr = torch.zeros_like(bin)
            bhz = torch.zeros_like(bhr)
            bhn = torch.zeros_like(bhz)
            input_tensors[input_start_index + 1] = self.create_attr_tensor(
                torch.zeros(input_tensors[input_start_index].shape[0], dtype=torch.float32)
            )
            input_tensors[input_start_index + 3] = self.create_attr_tensor(
                torch.zeros(input_tensors[input_start_index + 2].shape[0], dtype=torch.float32)
            )

            b_i_list = [self.create_attr_tensor(bir), self.create_attr_tensor(biz), self.create_attr_tensor(bin)]
            b_r_list = [self.create_attr_tensor(bhr), self.create_attr_tensor(bhz), self.create_attr_tensor(bhn)]

        return w_i_list, w_r_list, b_i_list, b_r_list

    def gru_hidden_state_helper(
        self,
        input_tensors,
        hidden_state_tensor,
        input_index,
        num_directions,
        direction_idx,
        num_layers,
        layer_idx,
        suffix,
        state_type,
        tf_state_tensors,
    ):

        tf_state_tensor = tf_state_tensors[0]
        assert hidden_state_tensor.dim() == 3
        slice_idx = layer_idx * num_directions + direction_idx
        if tf_state_tensor[slice_idx] is None:
            input_tensors[input_index] = self.create_attr_tensor(hidden_state_tensor[slice_idx])
            input_tensors[input_index].is_variable = True
        else:
            assert self.unroll_rnn, "Input state tensors are only supported when unroll_rnn=True is specified"
            input_tensors[input_index] = tf_state_tensor[slice_idx]

    def parse_common(
        self,
        input_tensor,
        hidden_state_tensor,
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
        self.unroll_rnn = True

        expected_num_params = 2 * num_layers
        params_step = 2
        if has_biases:
            expected_num_params *= 2
            params_step *= 2
        if bidirectional:
            expected_num_params *= 2

        assert (
            len(params_tensors) == expected_num_params
        ), f'num of params in GRU is wrong. got: {len(params_tensors)}, expected: {expected_num_params}'

        num_input_tensors = 7
        num_directions = 1
        state_start_index = [1, 8]

        if bidirectional:
            num_input_tensors *= 2
            num_directions *= 2

        suffixes = ["_fw", "_bw"]
        state_kinds = ["hidden"]
        param_start_indices = [0, params_step]
        input_start_indices = [2, 9]

        ops = []

        name = self.input_names[1]
        tf_in_state_tensors = [graph_converter.tensor_map.get(n, None) for n in name]
        tf_in_state_tensors = [
            self.find_or_create_input(1, graph_converter) if name in graph_converter.tensor_map else None
        ]

        tf_state_tensors = []
        unpacked_tensors = {}

        for t in tf_in_state_tensors:
            if t is not None and self.unroll_rnn:
                tensors = [
                    self.create_transform_tensor(np.squeeze(x, 0))
                    for x in np.split(t.tensor, num_directions * num_layers, 0)
                ]
                tf_state_tensors.append(tensors)
                ops.append(tfl.UnpackOperator([t], tensors, len(tensors), 0))
            else:
                tf_state_tensors.append([None] * num_directions * num_layers)

        current_input = self.find_or_create_input(0, graph_converter)
        gru_output = self.to_tfl_tensors(self.output_names[:1], self.output_tensors[:1])[0]

        params_offset = 0
        tf_out_state_tensors = [[]]

        for layer_idx in range(num_layers):
            inputs = [current_input] + [tfl.OptionalTensorInstance] * (num_input_tensors - 1)

            for direction_idx in range(num_directions):
                w_i_list, w_r_list, b_i_list, b_r_list = self.gru_input_helper(
                    inputs,
                    params_tensors,
                    has_biases,
                    params_offset + param_start_indices[direction_idx],
                    input_start_indices[direction_idx],
                    layer_idx,
                    suffixes[direction_idx],
                )

                self.gru_hidden_state_helper(
                    inputs,
                    hidden_state_tensor,
                    state_start_index[direction_idx],
                    num_directions,
                    direction_idx,
                    num_layers,
                    layer_idx,
                    suffixes[direction_idx],
                    state_kinds[0],
                    tf_state_tensors,
                )

            if layer_idx == num_layers - 1:
                layer_output = gru_output
            else:
                output_shape = list(input_tensor.shape)
                output_shape[-1] = inputs[4].shape[0] * num_directions
                layer_output = self.create_transform_tensor(np.empty(output_shape, dtype=inputs[0].dtype))

            outputs = [layer_output]

            if self.unroll_rnn:
                ts_axis = 1 if batch_first else 0
                num_timestep = inputs[0].shape[ts_axis]
                if inputs[0].name in unpacked_tensors:
                    input_ts = unpacked_tensors[inputs[0].name]
                else:
                    input_ts = [
                        self.create_transform_tensor(np.squeeze(x, ts_axis))
                        for x in np.split(inputs[0].tensor, num_timestep, ts_axis)
                    ]
                    ops.append(tfl.UnpackOperator([inputs[0]], input_ts, num_timestep, ts_axis))
                strides = [1, -1]

                output_ts = []

                for direction_idx in range(num_directions):

                    w_i_list, w_r_list, b_i_list, b_r_list = self.gru_input_helper(
                        inputs,
                        params_tensors,
                        has_biases,
                        params_offset + param_start_indices[direction_idx],
                        input_start_indices[direction_idx],
                        layer_idx,
                        suffixes[direction_idx],
                    )

                    state_start = state_start_index[direction_idx]
                    h = inputs[state_start]

                    stride = strides[direction_idx]

                    # Skip some computations for the first timestep
                    compute_h = h.buffer is None or np.any(h.tensor)

                    stacked_hs = []

                    for i, t in enumerate(input_ts[::stride]):

                        input_mm_list = []

                        if not self.separated_rnn_gate_calc:

                            wir, wiz, win = w_i_list
                            whr, whz, whn = w_r_list
                            bir, biz, bin = b_i_list
                            bhr, bhz, bhn = b_r_list

                            w_i = self.create_attr_tensor(np.concatenate([wir.tensor, wiz.tensor, win.tensor], 0))
                            w_h = self.create_attr_tensor(np.concatenate([whr.tensor, whz.tensor, whn.tensor], 0))
                            b_i = self.create_attr_tensor(np.concatenate([bir.tensor, biz.tensor, bin.tensor], 0))
                            b_h = self.create_attr_tensor(np.concatenate([bhr.tensor, bhz.tensor, bhn.tensor], 0))

                            input_mm = self.create_transform_tensor(
                                np.matmul(t.tensor, np.transpose(w_i.tensor, [1, 0])) + b_i.tensor
                            )
                            hidden_mm = self.create_transform_tensor(
                                np.matmul(h.tensor, np.transpose(w_h.tensor, [1, 0])) + b_h.tensor
                            )

                            ops.append(tfl.FullyConnectedOperator([t, w_i, b_i], [input_mm]))
                            ops.append(tfl.FullyConnectedOperator([h, w_h, b_h], [hidden_mm]))

                            left_in = np.split(input_mm.tensor, 3, axis=1)
                            dim_tensor = self.create_attr_tensor(np.array(1, dtype='int32'))
                            splited_left_in = [self.create_transform_tensor(t) for t in left_in]

                            ops.append(tfl.SplitOperator([dim_tensor, input_mm], splited_left_in, 3))

                            right_in = np.split(hidden_mm.tensor, 3, axis=-1)
                            splited_right_in = [self.create_transform_tensor(t) for t in right_in]

                            ops.append(tfl.SplitOperator([dim_tensor, hidden_mm], splited_right_in, 3))

                            rgate_left_in, zgate_left_in, ngate_left_in = splited_left_in
                            rgate_right_in, zgate_right_in, ngate_right_in_b = splited_right_in

                            rgate_in = self.create_transform_tensor(rgate_left_in.tensor + rgate_right_in.tensor)
                            ops.append(tfl.AddOperator([rgate_left_in, rgate_right_in], [rgate_in]))

                            rgate_out = self.create_transform_tensor(
                                torch.sigmoid(torch.from_numpy(rgate_in.tensor)).numpy()
                            )
                            ops.append(tfl.LogisticOperator([rgate_in], [rgate_out]))

                            zgate_in = self.create_transform_tensor(zgate_left_in.tensor + zgate_right_in.tensor)
                            ops.append(tfl.AddOperator([zgate_left_in, zgate_right_in], [zgate_in]))

                            zgate_out = self.create_transform_tensor(
                                torch.sigmoid(torch.from_numpy(zgate_in.tensor)).numpy()
                            )
                            ops.append(tfl.LogisticOperator([zgate_in], [zgate_out]))

                            ngate_right_in = self.create_transform_tensor(rgate_out.tensor * ngate_right_in_b.tensor)
                            ops.append(tfl.MulOperator([rgate_out, ngate_right_in_b], [ngate_right_in]))

                            ngate_in = self.create_transform_tensor(ngate_left_in.tensor + ngate_right_in.tensor)
                            ops.append(tfl.AddOperator([ngate_left_in, ngate_right_in], [ngate_in]))

                            ngate_out = self.create_transform_tensor(
                                torch.tanh(torch.from_numpy(ngate_in.tensor)).numpy()
                            )
                            ops.append(tfl.TanhOperator([ngate_in], [ngate_out]))

                            constant_tensor = self.create_attr_tensor(torch.tensor(1, dtype=torch.float32))

                            h_left_0 = self.create_transform_tensor(constant_tensor.tensor - zgate_out.tensor)
                            ops.append(tfl.SubOperator([constant_tensor, zgate_out], [h_left_0]))

                            h_left = self.create_transform_tensor(h_left_0.tensor * ngate_out.tensor)
                            ops.append(tfl.MulOperator([h_left_0, ngate_out], [h_left]))

                            if i != 0 or compute_h:
                                h_right = self.create_transform_tensor(zgate_out.tensor * h.tensor)
                                ops.append(tfl.MulOperator([zgate_out, h], [h_right]))

                                h = self.create_transform_tensor(h_left.tensor + h_right.tensor)
                                ops.append(tfl.AddOperator([h_left, h_right], [h]))

                            elif i == 0 and not compute_h:
                                h = h_left

                            stacked_hs.append(h)

                        else:
                            for j, (w_i, b_i) in enumerate(zip(w_i_list, b_i_list)):

                                input_mm = self.create_transform_tensor(
                                    np.matmul(t.tensor, np.transpose(w_i.tensor, [1, 0])) + b_i.tensor
                                )
                                ops.append(tfl.FullyConnectedOperator([t, w_i, b_i], [input_mm]))
                                input_mm_list.append(input_mm)

                            if i != 0 or compute_h:

                                hidden_mm_list = []
                                for j, (w_r, b_r) in enumerate(zip(w_r_list, b_r_list)):

                                    hidden_mm = self.create_transform_tensor(
                                        np.matmul(h.tensor, np.transpose(w_r.tensor, [1, 0])) + b_r.tensor
                                    )
                                    ops.append(tfl.FullyConnectedOperator([h, w_r, b_r], [hidden_mm]))
                                    hidden_mm_list.append(hidden_mm)
                            else:
                                hidden_mm_list = b_r_list

                            # calculate r,z,n gates
                            rgate_in = self.create_transform_tensor(input_mm_list[0].tensor + hidden_mm_list[0].tensor)
                            ops.append(tfl.AddOperator([input_mm_list[0], hidden_mm_list[0]], [rgate_in]))

                            zgate_in = self.create_transform_tensor(input_mm_list[1].tensor + hidden_mm_list[1].tensor)
                            ops.append(tfl.AddOperator([input_mm_list[1], hidden_mm_list[1]], [zgate_in]))

                            zgate_out = self.create_transform_tensor(
                                torch.sigmoid(torch.from_numpy(zgate_in.tensor)).numpy()
                            )
                            ops.append(tfl.LogisticOperator([zgate_in], [zgate_out]))

                            rgate_out = self.create_transform_tensor(
                                torch.sigmoid(torch.from_numpy(rgate_in.tensor)).numpy()
                            )
                            ops.append(tfl.LogisticOperator([rgate_in], [rgate_out]))

                            ngate_in_hside = self.create_transform_tensor(rgate_out.tensor * hidden_mm_list[2].tensor)
                            ops.append(tfl.MulOperator([rgate_out, hidden_mm_list[2]], [ngate_in_hside]))

                            ngate_in = self.create_transform_tensor(input_mm_list[2].tensor + ngate_in_hside.tensor)
                            ops.append(tfl.AddOperator([input_mm_list[2], ngate_in_hside], [ngate_in]))

                            ngate_out = self.create_transform_tensor(
                                torch.tanh(torch.from_numpy(ngate_in.tensor)).numpy()
                            )
                            ops.append(tfl.TanhOperator([ngate_in], [ngate_out]))

                            constant_tensor = self.create_attr_tensor(torch.tensor(1, dtype=torch.float32))

                            h_left_0 = self.create_transform_tensor(constant_tensor.tensor - zgate_out.tensor)
                            ops.append(tfl.SubOperator([constant_tensor, zgate_out], [h_left_0]))

                            h_left = self.create_transform_tensor(h_left_0.tensor * ngate_out.tensor)
                            ops.append(tfl.MulOperator([h_left_0, ngate_out], [h_left]))

                            if i != 0 or compute_h:
                                h_right = self.create_transform_tensor(zgate_out.tensor * h.tensor)
                                ops.append(tfl.MulOperator([zgate_out, h], [h_right]))

                                h = self.create_transform_tensor(h_left.tensor + h_right.tensor)
                                ops.append(tfl.AddOperator([h_left, h_right], [h]))

                            elif i == 0 and not compute_h:
                                h = h_left

                            stacked_hs.append(h)

                    tf_out_state_tensors[0].append(h)
                    output_ts.extend(stacked_hs[::stride])

                if bidirectional:
                    fw_out = self.create_transform_tensor(
                        np.stack([x.tensor for x in output_ts[:num_timestep]], ts_axis)
                    )
                    ops.append(tfl.PackOperator(output_ts[:num_timestep], [fw_out], num_timestep, axis=ts_axis))

                    bw_out = self.create_transform_tensor(
                        np.stack([x.tensor for x in output_ts[:num_timestep]], ts_axis)
                    )
                    ops.append(tfl.PackOperator(output_ts[num_timestep:], [bw_out], num_timestep, axis=ts_axis))

                    ops.append(tfl.ConcatenationOperator([fw_out, bw_out], outputs, axis=2))

                elif layer_idx != num_layers - 1:
                    # Reusing unpacked tensors for the logic in the next layer
                    unpacked_tensors[outputs[0].name] = output_ts
                else:
                    # For the last layer, we have to pack the together
                    ops.append(tfl.PackOperator(output_ts, outputs, len(output_ts), axis=ts_axis))

            current_input = outputs[0]
            params_offset += params_step * num_directions

        if self.unroll_rnn:
            state_outputs = self.to_tfl_tensors(self.output_names[1:], self.output_tensors[1:])
            for i, (orig, new) in enumerate(zip(tf_in_state_tensors, tf_out_state_tensors)):
                if orig is not None:
                    pack_op = tfl.PackOperator(new, state_outputs[i : i + 1], len(new), 0)
                    pack_op.extra_hints['warn_on_unused'] = False
                    ops.append(pack_op)

        for op in ops:
            graph_converter.add_operator(op)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor, hidden_state_tensor, params_tensors = self.input_tensors[:3]
        has_biases, num_layers, dropout, is_train, bidirectional, batch_first = self.input_tensors[3:]

        self.parse_common(
            input_tensor,
            hidden_state_tensor,
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
        ), "Running mean and variance should not be None for aten::batch_norm. Otherwise, use LayerNorm instead."

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

        assert (
            divisor_override is None or divisor_override == kernel_h == kernel_w
        ), "Only divisor_override == kernel_h == kernel_w is supported"

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
        assert all(x == 1.0 for x in self.input_tensors[1:]), "Only alpha == scale == input_scale == 1 is supported"
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
        assert alpha == 1, "Only alpha == 1 is supported"

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif not isinstance(other, torch.Tensor):
            assert False, "other should have type int, float, tensor in aten::sub(input, other)"

        self.elementwise_binary(tfl.SubOperator, graph_converter, True)


class ATenRsubOperator(ATenRsubSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        other = self.input_tensors[1]
        alpha = self.input_tensors[-1]
        assert alpha == 1, "Only alpha == 1 is supported"

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif not isinstance(other, torch.Tensor):
            assert False, "other should have type int, float, tensor in aten::rsub(input, other)"

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
        elif not isinstance(other, torch.Tensor):
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
        elif not isinstance(other, torch.Tensor):
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
        assert self.input_tensors[0].dtype in (
            torch.float32,
            torch.int32,
        ), "Input should be tensors of type torch.float32 or torch.int32"

        if not isinstance(self.input_tensors[1], torch.Tensor):
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

        assert dilation_h == dilation_w == 1, "Only dilation == 1 is supported"

        add_pad_op = not (
            stride_h == stride_w == 1 and pad_h == kernel_h // 2 and pad_w == kernel_w // 2 and not ceil_mode
        )
        padding = tfl_schema.Padding.SAME
        if add_pad_op:
            padding = tfl_schema.Padding.VALID

        maxpool_op = tfl.MaxPool2dOperator(inputs, outputs, padding, stride_w, stride_h, kernel_w, kernel_h)
        ops = self.wrap_ops_with_nhwc_nchw_transposes([maxpool_op])
        if add_pad_op:
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
        assert type(dim) is int

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
        assert type(dim) is int

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

        update_name = None
        if weight_c == input_c:
            new_alpha = self.create_transform_tensor(np.reshape(alpha_tensor.tensor, new_shape))
            graph_converter.add_operator(tfl.ReshapeOperator([alpha_tensor, shape_tensor], [new_alpha], new_shape))
        elif input_c != weight_c:
            new_alpha = self.create_transform_tensor(np.tile(alpha_tensor.tensor, new_shape))
            if alpha_tensor.buffer is None:
                graph_converter.add_operator(tfl.TileOperator([alpha_tensor, shape_tensor], [new_alpha]))
            else:
                store = graph_converter.get_transform_store(alpha_tensor.name, str(input_c))
                if store is None:
                    graph_converter.add_transform_store(alpha_tensor.name, str(input_c), new_alpha.name)
                    update_name = new_alpha.name
                    new_alpha = new_alpha.tensor
                else:
                    update_name = store

        self.input_tensors[1] = new_alpha
        if update_name is None:
            self.input_names[1] = new_alpha.name
        else:
            self.input_names[1] = update_name

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
        if not isinstance(self.input_tensors[1], torch.Tensor):
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

        if len(stride) != len(padding) and len(stride) == 1:
            stride = stride * len(padding)

        if transpose == 0:
            graph_converter.add_operator(
                tfl.GenericConvOperator(inputs, outputs, stride, padding, dilation, output_padding, groups)
            )
        else:
            graph_converter.add_operator(
                tfl.GenericTransposeConvOperator(
                    inputs,
                    outputs,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    groups,
                    self.enable_mtk_ops,
                    self.conv_transpose_with_bias,
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

        if self.input_names[2] in graph_converter.constant_mapping:
            start_t = graph_converter.constant_mapping[self.input_names[2]]
            new_shape_arr = np.array((1,), dtype='int32')
            new_shape_tensor = self.create_attr_tensor(new_shape_arr)
            start_reshaped = self.create_transform_tensor(np.reshape(start_t.tensor, new_shape_arr))
            graph_converter.add_operator(
                tfl.ReshapeOperator([start_t, new_shape_tensor], [start_reshaped], new_shape_arr)
            )

            start_casted = self.create_transform_tensor(start_reshaped.tensor.astype('int32'))
            graph_converter.add_operator(
                tfl.CastOperator(
                    [start_reshaped],
                    [start_casted],
                    tfl.numpy_tflite_dtype_mappings[str(start_reshaped.dtype)],
                    tfl.numpy_tflite_dtype_mappings[str(start_casted.dtype)],
                )
            )

            start_tensor = self.create_transform_tensor(starts)
            starts_left = starts[:dim]
            starts_right = starts[dim + 1 :]
            starts_tensors = []
            if len(starts_left) > 0:
                starts_tensors.append(self.create_attr_tensor(starts_left))
            starts_tensors.append(start_casted)
            if len(starts_right) > 0:
                starts_tensors.append(self.create_attr_tensor(starts_right))
            if len(starts_tensors) > 1:
                graph_converter.add_operator(tfl.ConcatenationOperator(starts_tensors, [start_tensor], 0))
            else:
                start_tensor = starts_tensors[0]
        else:
            start_tensor = self.create_attr_tensor(starts)

        ends = np.array(input_tensor.tensor.shape, dtype='int32')
        if step != 1 or start_tensor.buffer is None or self.input_names[3] in graph_converter.constant_mapping:
            ends[dim] = end
        else:
            ends[dim] = end - start

        if self.input_names[3] in graph_converter.constant_mapping:
            end_t = graph_converter.constant_mapping[self.input_names[3]]
            new_shape_arr = np.array((1,), dtype='int32')
            new_shape_tensor = self.create_attr_tensor(new_shape_arr)
            end_reshaped = self.create_transform_tensor(np.reshape(end_t.tensor, new_shape_arr))
            graph_converter.add_operator(tfl.ReshapeOperator([end_t, new_shape_tensor], [end_reshaped], new_shape_arr))

            end_casted = self.create_transform_tensor(end_reshaped.tensor.astype('int32'))
            graph_converter.add_operator(
                tfl.CastOperator(
                    [end_reshaped],
                    [end_casted],
                    tfl.numpy_tflite_dtype_mappings[str(end_reshaped.dtype)],
                    tfl.numpy_tflite_dtype_mappings[str(end_casted.dtype)],
                )
            )

            end_tensor = self.create_transform_tensor(ends)
            ends_left = ends[:dim]
            ends_right = ends[dim + 1 :]
            ends_tensors = []
            if len(ends_left) > 0:
                ends_tensors.append(self.create_attr_tensor(ends_left))
            ends_tensors.append(end_casted)
            if len(ends_right) > 0:
                ends_tensors.append(self.create_attr_tensor(ends_right))
            if len(ends_tensors) > 1:
                graph_converter.add_operator(tfl.ConcatenationOperator(ends_tensors, [end_tensor], 0))
            else:
                end_tensor = ends_tensors[0]
        else:
            end_tensor = self.create_attr_tensor(ends)

        if step != 1 or start_tensor.buffer is None or end_tensor.buffer is None:
            strides = np.ones(input_tensor.tensor.ndim, dtype='int32')
            strides[dim] = step

            stride_tensor = self.create_attr_tensor(strides)

            inputs = [input_tensor, start_tensor, end_tensor, stride_tensor]
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            graph_converter.add_operator(tfl.StridedSliceOperator(inputs, outputs))
        else:
            size_tensor = end_tensor
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
        assert alpha == 1, "Only alpha == 1 is supported"

        if type(other) in (int, float, bool):
            self.input_tensors[1] = torch.tensor([other], dtype=self.input_tensors[0].dtype)
        elif not isinstance(other, torch.Tensor):
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

        if self.q_type == np.int16:
            self.output_tensors[0] = torch.quantize_per_tensor(
                self.output_tensors[0].dequantize(),
                self.output_tensors[0].q_scale() * 2,
                0,
                self.output_tensors[0].dtype,
            )

        self.elementwise_unary(tfl.LogisticOperator, graph_converter)


class ATenSelectOperator(ATenSelectSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor = self.find_or_create_input(0, graph_converter)
        dim, index = self.input_tensors[1:]

        assert type(dim) is int
        assert type(index) is int

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

        assert weight.tensor.ndim == 2, "Only 2D weight tensors are supported"
        assert indices.dtype in (np.int32, np.int64), "Only integral indices are supported"

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
        if type(self) is ATenClampOperator:
            min_value, max_value = self.input_tensors[1:]
        elif type(self) is ATenClampMinOperator:
            min_value, max_value = self.input_tensors[1], None
        elif type(self) is ATenClampMaxOperator:
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
            if has_min:
                if input_tensor.quantization is not None:
                    min_value_arr = np.array([min_value], dtype='float32')
                    min_value_tensor = self.create_attr_tensor(
                        self.quantize_numpy(
                            min_value_arr,
                            input_tensor.quantization.scale,
                            input_tensor.quantization.zero_point,
                            input_tensor.dtype,
                        ),
                        quantization=input_tensor.quantization,
                    )
                else:
                    min_value_arr = np.array([min_value], dtype=input_tensor.dtype)
                    min_value_tensor = self.create_attr_tensor(min_value_arr)
            if has_max:
                if input_tensor.quantization is not None:
                    max_value_arr = np.array([max_value], dtype='float32')
                    max_value_tensor = self.create_attr_tensor(
                        self.quantize_numpy(
                            max_value_arr,
                            input_tensor.quantization.scale,
                            input_tensor.quantization.zero_point,
                            input_tensor.dtype,
                        ),
                        quantization=input_tensor.quantization,
                    )
                else:
                    max_value_arr = np.array([max_value], dtype=input_tensor.dtype)
                    max_value_tensor = self.create_attr_tensor(max_value_arr)
            if has_min and has_max:
                inter_tensor = self.create_transform_tensor(
                    np.minimum(input_tensor.tensor, min_value_tensor.tensor), quantization=input_tensor.quantization
                )
                ops.append(tfl.MaximumOperator([input_tensor, min_value_tensor], [inter_tensor]))

                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                ops.append(tfl.MinimumOperator([inter_tensor, max_value_tensor], outputs))
            elif has_min:
                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
                ops.append(tfl.MaximumOperator([input_tensor, min_value_tensor], outputs))
            else:
                outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
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
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.NotEqualOperator, graph_converter, True)


class ATenSoftplusOperator(ATenSoftplusSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        beta = self.input_tensors[1]

        assert beta == 1.0, "Only beta=1.0 is supported for aten::softplus"
        warnings.warn('threshold is ignored when transforming aten::softplus')

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


class ATenGroupNormOperator(ATenGroupNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        inp = self.find_or_create_input(0, graph_converter)
        eps = self.input_tensors[args['eps']]

        n_channels = inp.shape[1]
        n_groups, weight, bias = self.input_tensors[1:4]
        affine = False
        if weight is not None and bias is not None:
            affine = True
            weight, bias = [self.find_or_create_input(i, graph_converter) for i in range(2, 4)]

        ops = []

        inputs = []
        dims = len(inp.shape)
        if n_channels == n_groups and n_groups > 1:
            axis = tuple(range(2, dims))
            axis_tensor = self.create_attr_tensor(np.array(axis, dtype='int32'))
            inputs.append(inp)
        elif n_groups == 1:
            axis = tuple(range(1, dims))
            axis_tensor = self.create_attr_tensor(np.array(axis, dtype='int32'))
            inputs.append(inp)
        else:
            axis = tuple(range(1, dims))
            axis_tensor = self.create_attr_tensor(np.array(axis, dtype='int32'))
            split_dim_tensor = self.create_attr_tensor(np.array(1, dtype='int32'))
            inputs = [self.create_transform_tensor(t) for t in np.split(inp.tensor, n_groups, axis=1)]
            ops.append(tfl.SplitOperator([split_dim_tensor, inp], inputs, n_groups))

        dim_ones = (1,) * (dims - 2)

        norms = []
        for input_t in inputs:
            mean = self.create_transform_tensor(np.mean(input_t.tensor, axis=axis, keepdims=True))
            ops.append(tfl.MeanOperator([input_t, axis_tensor], [mean], keepDims=True))

            squared_diff = self.create_transform_tensor(np.power(input_t.tensor - mean.tensor, 2))
            ops.append(tfl.SquaredDifferenceOperator([input_t, mean], [squared_diff]))

            var = self.create_transform_tensor(np.mean(squared_diff.tensor, axis=axis, keepdims=True))
            ops.append(tfl.MeanOperator([squared_diff, axis_tensor], [var], keepDims=True))

            numerator = self.create_transform_tensor(input_t.tensor - mean.tensor)
            ops.append(tfl.SubOperator([input_t, mean], [numerator]))

            eps_tensor = self.create_attr_tensor(np.array([eps], dtype='float32'))
            with_eps = self.create_transform_tensor(var.tensor + eps_tensor.tensor)
            ops.append(tfl.AddOperator([var, eps_tensor], [with_eps]))

            denominator = self.create_transform_tensor(np.sqrt(with_eps.tensor))
            ops.append(tfl.SqrtOperator([with_eps], [denominator]))

            norm = self.create_transform_tensor(numerator.tensor / denominator.tensor)
            ops.append(tfl.DivOperator([numerator, denominator], [norm]))

            norms.append(norm)

        if len(norms) > 1:
            cat_out = self.create_transform_tensor(np.concatenate([x.tensor for x in norms], 1))
            ops.append(tfl.ConcatenationOperator(norms, [cat_out], 1))
        else:
            cat_out = norms[0]

        if affine:
            weight.tensor = weight.tensor.reshape(-1, *dim_ones)
            bias.tensor = bias.tensor.reshape(-1, *dim_ones)

            weight_tensor = self.create_attr_tensor(weight.tensor)
            bias_tensor = self.create_attr_tensor(bias.tensor)

            mul_out = self.create_transform_tensor(cat_out.tensor * weight_tensor.tensor)
            ops.append(tfl.MulOperator([cat_out, weight_tensor], [mul_out]))

            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            ops.append(tfl.AddOperator([mul_out, bias_tensor], outputs))
        else:
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
            ops[-1].outputs = outputs

        for op in ops:
            graph_converter.add_operator(op)


class ATenIndexOperator(ATenIndexSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        indices = self.input_tensors[1]

        filtered_dims = [i for i, idx in enumerate(indices) if idx is not None]
        assert all((indices[i].dtype in (torch.int64, torch.int32) for i in filtered_dims))

        input_tensor = self.find_or_create_input(0, graph_converter)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if len(filtered_dims) > 1:
            if graph_converter.has_nested_names(self.input_names[1]):
                input_names = graph_converter.get_list_expanded_names(self.input_names[1])
                indices_tensors = self.to_tfl_tensors(
                    input_names, self.input_tensors[1], graph_converter=graph_converter, non_existent_as_buffer=True
                )
            else:
                if type(self.input_tensors[1]) in (tuple, list):
                    indices_tensors = [self.create_attr_tensor(x) for x in self.input_tensors[1]]
                else:
                    indices_tensors = [self.find_or_create_input(1, graph_converter)]

            dim = input_tensor.tensor.ndim

            indices_shape = [x.tensor.size for x in indices_tensors]
            max_len = max(indices_shape)
            indices_shape_tensor = torch.tensor(indices_shape)
            left_indices = (
                torch.arange(max_len).view(-1, 1).expand(-1, len(indices_shape)) % indices_shape_tensor
            ).int()
            all_indices_shape = list(outputs[0].shape) + [dim]

            if len(indices_tensors) < dim:
                pad_shape = list(input_tensor.shape[len(indices_tensors) :])
                pad_indices = torch.ones(pad_shape).nonzero().int()
                left_len = len(indices_shape)
                right_len = len(pad_shape)
                left_size = left_indices.size(0)
                right_size = pad_indices.size(0)
                left_reshaped = (
                    left_indices.view(-1, 1, left_len).expand(-1, right_size, left_len).reshape(-1, left_len)
                )
                right_reshaped = (
                    pad_indices.view(1, -1, right_len).expand(left_size, -1, right_len).reshape(-1, right_len)
                )
                all_indices = torch.cat([left_reshaped, right_reshaped], 1).view(all_indices_shape).unbind(-1)
            else:
                all_indices = left_indices.view(all_indices_shape).unbind(-1)

            new_indices = []
            for i in range(dim):
                if i < len(indices_tensors):
                    idx_tensor = indices_tensors[i]
                    actual_idx = np.take(idx_tensor.tensor, all_indices[i].numpy())
                else:
                    actual_idx = all_indices[i].numpy()
                if idx_tensor.buffer is None and i < len(indices_tensors):
                    actual_idx_t = self.create_transform_tensor(actual_idx)
                    fake_idx_t = self.create_attr_tensor(all_indices[i].numpy())
                    graph_converter.add_operator(tfl.GatherOperator([idx_tensor, fake_idx_t], [actual_idx_t], axis=0))

                    if str(actual_idx_t.dtype) != 'int32':
                        index_casted = self.create_transform_tensor(actual_idx_t.tensor.astype('int32'))
                        graph_converter.add_operator(
                            tfl.CastOperator(
                                [actual_idx_t],
                                [index_casted],
                                tfl.numpy_tflite_dtype_mappings[str(actual_idx_t.dtype)],
                                tfl.numpy_tflite_dtype_mappings[str(index_casted.dtype)],
                            )
                        )
                        actual_idx_t = index_casted
                    new_indices.append(actual_idx_t)
                else:
                    new_indices.append(self.create_attr_tensor(actual_idx.astype(np.int32)))

            index_arr = np.stack([x.tensor for x in new_indices], -1)
            if all((x.buffer is not None for x in new_indices)):
                index_tensor = self.create_attr_tensor(index_arr)
            else:
                index_tensor = self.create_transform_tensor(index_arr)
                graph_converter.add_operator(
                    tfl.PackOperator(new_indices, [index_tensor], dim, axis=index_tensor.tensor.ndim - 1)
                )

            graph_converter.add_operator(tfl.GatherNdOperator([input_tensor, index_tensor], outputs))
        else:
            try:
                names = graph_converter.get_list_expanded_names(self.input_names[1])
            except KeyError:
                names = [self.get_unique_attr_name() for _ in indices]

            filtered_names = [names[i] for i in filtered_dims]
            filtered_tensors = [indices[i].to(dtype=torch.int32) for i in filtered_dims]

            filtered_tensors = [
                t + (t < 0).int() * input_tensor.shape[i] if n not in graph_converter.tensor_map else t
                for i, n, t in zip(filtered_dims, filtered_names, filtered_tensors)
            ]
            indice_tensors = self.to_tfl_tensors(
                filtered_names, filtered_tensors, graph_converter=graph_converter, non_existent_as_buffer=True
            )

            actual_input = input_tensor
            actual_output = None
            for i, (dim, idx) in enumerate(zip(filtered_dims, indice_tensors)):
                if i == len(filtered_dims) - 1:
                    actual_output = outputs[0]
                else:
                    actual_output = self.create_transform_tensor(np.take(actual_input.tensor, idx.tensor, axis=dim))

                graph_converter.add_operator(tfl.GatherOperator([actual_input, idx], [actual_output], axis=dim))

                actual_input = actual_output


class ATenIndexSelectOperator(ATenIndexSelectSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        dim = self.input_tensors[1]
        indices = self.input_tensors[2]

        assert indices.dtype in (torch.int64, torch.int32)

        input_tensor = self.find_or_create_input(0, graph_converter)

        if dim < 0:
            dim += len(input_tensor.shape)

        new_indices = indices.to(dtype=torch.int32)
        new_indices = new_indices + (new_indices < 0).int() * input_tensor.shape[dim]

        indices_tensor = self.to_tfl_tensors(
            self.input_names[2:3], [new_indices], graph_converter=graph_converter, non_existent_as_buffer=True
        )[0]

        self.create_attr_tensor(new_indices)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.GatherOperator([input_tensor, indices_tensor], outputs, axis=dim))


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


class ATenRepeatInterleaveOperator(ATenRepeatInterleaveSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        if 'dim' in args:
            dim = self.input_tensors[args['dim']]
        else:
            dim = None

        if 'repeats' in args:
            repeats = self.input_tensors[args['repeats']]
        else:
            repeats = None

        if repeats is None:
            size_repeats = input_tensor.tensor.size
            raw_indices = torch.arange(size_repeats, dtype=torch.int32)
            repeats_tensor = input_tensor
        elif type(repeats) is int:
            if dim is None:
                size_repeats = input_tensor.tensor.size
            else:
                size_repeats = input_tensor.shape[dim]
            raw_indices = torch.arange(size_repeats, dtype=torch.int32)
            repeats_arr = torch.tensor(repeats, dtype=torch.int32)
            repeats_tensor = self.create_attr_tensor(repeats_arr)
        else:
            if dim is None:
                size_repeats = input_tensor.tensor.size
            else:
                size_repeats = input_tensor.shape[dim]
            raw_indices = torch.arange(size_repeats, dtype=torch.int32)
            repeats_tensor = self.find_or_create_input(args['repeats'], graph_converter)

        assert repeats_tensor.buffer is not None, "dynamic repeats_tensor is not supported"

        actual_indices = self.create_attr_tensor(
            torch.repeat_interleave(raw_indices, torch.from_numpy(repeats_tensor.tensor).long())
        )

        actual_input = input_tensor
        if dim is None and len(input_tensor.shape) > 1:
            new_shape = (input_tensor.tensor.size,)
            shape_tensor = self.create_attr_tensor(np.array(new_shape, dtype='int32'))
            actual_input = self.create_transform_tensor(np.reshape(input_tensor.tensor, new_shape))
            graph_converter.add_operator(tfl.ReshapeOperator([input_tensor, shape_tensor], [actual_input], new_shape))

        inputs = [actual_input, actual_indices]
        gather_dim = dim
        if gather_dim is None:
            gather_dim = 0
        if gather_dim < 0:
            gather_dim += input_tensor.tensor.ndim
        graph_converter.add_operator(tfl.GatherOperator(inputs, outputs, gather_dim))


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

        dim_tensor = self.create_attr_tensor(np.array(dim, dtype='int32'))
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
        dim_tensor = self.create_attr_tensor(np.array(dim, dtype='int32'))

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
            fill_arr = np.zeros(indices_tensor.shape[:-1], dtype=input_tensor.dtype)
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


class ATenIndexPutOperator(ATenIndexPutSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        # torch.Tensor.index_put_ requires index tensor of type `torch.int64`
        accumulate = self.input_tensors[3]
        assert not accumulate, "aten::index_put_ with accumulate=True is not supported"

        orig_type = self.input_tensors[1][0].dtype
        self.input_tensors[1] = tuple([x.to(dtype=torch.int64) for x in self.input_tensors[1]])
        self.run(node)

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        self.input_tensors[1] = tuple([x.to(dtype=orig_type) for x in self.input_tensors[1]])

        if graph_converter.has_nested_names(self.input_names[1]):
            input_names = graph_converter.get_list_expanded_names(self.input_names[1])
            indices_tensors = self.to_tfl_tensors(
                input_names, self.input_tensors[1], graph_converter=graph_converter, non_existent_as_buffer=True
            )
        else:
            if type(self.input_tensors[1]) in (tuple, list):
                indices_tensors = [self.create_attr_tensor(x) for x in self.input_tensors[1]]
            else:
                indices_tensors = [self.find_or_create_input(1, graph_converter)]

        dim = input_tensor.tensor.ndim

        indices_shape = [x.tensor.size for x in indices_tensors]
        max_len = max(indices_shape)
        indices_shape_tensor = torch.tensor(indices_shape)
        left_indices = (torch.arange(max_len).view(-1, 1).expand(-1, len(indices_shape)) % indices_shape_tensor).int()

        if len(indices_tensors) < dim:
            pad_shape = list(input_tensor.shape[len(indices_tensors) :])
            pad_indices = torch.ones(pad_shape).nonzero().int()
            left_len = len(indices_shape)
            right_len = len(pad_shape)
            left_size = left_indices.size(0)
            right_size = pad_indices.size(0)
            left_reshaped = left_indices.view(-1, 1, left_len).expand(-1, right_size, left_len).reshape(-1, left_len)
            right_reshaped = pad_indices.view(1, -1, right_len).expand(left_size, -1, right_len).reshape(-1, right_len)
            all_indices = torch.cat([left_reshaped, right_reshaped], 1).unbind(1)
        else:
            all_indices = left_indices.unbind(1)

        new_indices = []
        for i in range(dim):
            if i < len(indices_tensors):
                idx_tensor = indices_tensors[i]
                actual_idx = np.take(idx_tensor.tensor, all_indices[i].numpy())
            else:
                actual_idx = all_indices[i].numpy()
            if idx_tensor.buffer is None and i < len(indices_tensors):
                actual_idx_t = self.create_transform_tensor(actual_idx)
                fake_idx_t = self.create_attr_tensor(all_indices[i].numpy())
                graph_converter.add_operator(tfl.GatherOperator([idx_tensor, fake_idx_t], [actual_idx_t], axis=0))

                if str(actual_idx_t.dtype) != 'int32':
                    index_casted = self.create_transform_tensor(actual_idx_t.tensor.astype('int32'))
                    graph_converter.add_operator(
                        tfl.CastOperator(
                            [actual_idx_t],
                            [index_casted],
                            tfl.numpy_tflite_dtype_mappings[str(actual_idx_t.dtype)],
                            tfl.numpy_tflite_dtype_mappings[str(index_casted.dtype)],
                        )
                    )
                    actual_idx_t = index_casted
                new_indices.append(actual_idx_t)
            else:
                new_indices.append(self.create_attr_tensor(actual_idx.astype(np.int32)))

        index_arr = np.stack([x.tensor for x in new_indices], 1)
        if all((x.buffer is not None for x in new_indices)):
            index_tensor = self.create_attr_tensor(index_arr)
        else:
            index_tensor = self.create_transform_tensor(index_arr)
            graph_converter.add_operator(tfl.PackOperator(new_indices, [index_tensor], dim, axis=1))

        val_tensor = self.find_or_create_input(2, graph_converter)
        actual_val = val_tensor
        orig_val_shape = val_tensor.shape
        target_val_shape = index_tensor.shape[:-1]
        if orig_val_shape != target_val_shape:
            if val_tensor.buffer is None:
                new_shape = orig_val_shape
                val_reshaped = val_tensor
                if len(target_val_shape) > len(orig_val_shape):
                    new_shape = [1] * (len(target_val_shape) - len(orig_val_shape)) + list(orig_val_shape)
                    new_shape_arr = np.array(new_shape, dtype='int32')
                    new_shape_tensor = self.create_attr_tensor(new_shape_arr)
                    reshaped = self.create_transform_tensor(np.reshape(val_tensor.tensor, new_shape_arr))
                    val_reshaped = reshaped
                    reshape_op = tfl.ReshapeOperator([val_tensor, new_shape_tensor], [reshaped], new_shape_arr)
                    reshape_op.extra_hints['direction'] = 'up'
                    graph_converter.add_operator(reshape_op)

                repeats = []
                for x, y in zip(new_shape, target_val_shape):
                    if x != y:
                        repeats.append(y // x)
                    else:
                        repeats.append(1)

                actual_val = self.create_transform_tensor(np.tile(val_reshaped.tensor, repeats))
                repeat_tensor = self.create_attr_tensor(np.array(repeats, dtype='int32'))
                graph_converter.add_operator(tfl.TileOperator([val_reshaped, repeat_tensor], [actual_val]))
            else:
                actual_val = self.create_attr_tensor(np.broadcast_to(val_tensor.tensor, target_val_shape))

        shape_tensor = self.create_attr_tensor(np.array(input_tensor.shape, dtype='int32'))

        if input_tensor.buffer is None or index_tensor.buffer is None:
            old_val_tensor = self.create_transform_tensor(actual_val.tensor)
            graph_converter.add_operator(tfl.GatherNdOperator([input_tensor, index_tensor], [old_val_tensor]))
        else:
            transformed_index = tuple(index_tensor.tensor[..., i] for i in range(index_tensor.shape[-1]))
            old_val_tensor = self.create_attr_tensor(input_tensor.tensor[transformed_index])

        if actual_val.buffer is None:
            update_tensor = self.create_transform_tensor(actual_val.tensor - old_val_tensor.tensor)
            graph_converter.add_operator(tfl.SubOperator([actual_val, old_val_tensor], [update_tensor]))
        else:
            update_tensor = self.create_attr_tensor(actual_val.tensor - old_val_tensor.tensor)

        updated_tensor = self.create_transform_tensor(input_tensor.tensor)
        graph_converter.add_operator(
            tfl.ScatterNdOperator([index_tensor, update_tensor, shape_tensor], [updated_tensor])
        )

        graph_converter.add_operator(tfl.AddOperator([input_tensor, updated_tensor], [output_tensor]))


class ATenGeluOperator(ATenGeluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        ops = []

        input_tensor = self.find_or_create_input(0, graph_converter)
        output_tensor = self.to_tfl_tensors(self.output_names, self.output_tensors)[0]

        approximate = "none"
        if 'approximate' in args:
            approximate = self.input_tensors[args['approximate']] or "none"

        if self.legacy_gelu:
            if approximate == "none":
                warnings.warn('aten::gelu[approximate="none"] is not supported with legacy_gelu=True')

            constant_tensor = self.create_attr_tensor(np.array([1.702], dtype='float32'))
            sigmoid_in = self.create_transform_tensor(input_tensor.tensor * constant_tensor.tensor)

            actual_input = input_tensor
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
        else:
            op = tfl.GeluOperator([input_tensor], [output_tensor], approximate == "none")
            ops.append(op)

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
        if not isinstance(self.input_tensors[1], torch.Tensor):
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

        if 'other' in args:
            self.elementwise_binary(tfl.MinimumOperator, graph_converter, True)
        else:
            self.handle_reduce(tfl.ReduceMinOperator, args, graph_converter, False)


class ATenMaxOperator(ATenMaxSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        if 'other' in args:
            self.elementwise_binary(tfl.MaximumOperator, graph_converter, True)
        else:
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
        dim_tensor = self.create_attr_tensor(np.array(dim, dtype='int32'))
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
            if type(t) is torch.Tensor:
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
        if type(other) is torch.Tensor:
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


class ATenMaximumOperator(ATenMaximumSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.MaximumOperator, graph_converter, True)


class ATenMinimumOperator(ATenMinimumSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.MinimumOperator, graph_converter, True)


class ATenGtOperator(ATenGtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.GreaterOperator, graph_converter, True)


class ATenLtOperator(ATenLtSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.LessOperator, graph_converter, True)


class ATenGeOperator(ATenGeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.GreaterEqualOperator, graph_converter, np.True_)


class ATenLeOperator(ATenLeSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = self.torch_tensor_from_scalar(self.input_tensors[0], self.input_tensors[1])

        self.elementwise_binary(tfl.LessEqualOperator, graph_converter, True)


class ATenRemainderOperator(ATenRemainderSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        if not isinstance(self.input_tensors[1], torch.Tensor):
            self.input_tensors[1] = torch.tensor([self.input_tensors[1]], dtype=self.input_tensors[0].dtype)

        self.elementwise_binary(tfl.FloorModOperator, graph_converter, True)


class ATenWhereOperator(ATenWhereSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        assert 'self' in args and 'other' in args, "aten::where(condition) is not supported"

        if not isinstance(self.input_tensors[2], torch.Tensor):
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

        if str(input_tensor.dtype) == 'int64' and input_tensor.tensor.ndim == 1 and input_tensor.tensor.size == 1:
            shape_tensor = self.create_attr_tensor(np.array((), dtype='int32'))
            graph_converter.add_operator(tfl.ReshapeOperator([input_tensor, shape_tensor], outputs, []))
        else:
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
                dim_tensor = self.create_attr_tensor(np.array(dim, dtype='int32'))
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
    def parse_common(self, node, attrs, args, graph_converter):
        p = self.input_tensors[1]
        assert p in (1, 2), "only torch.norm with p=1,2 is supported"

        input_t = self.find_or_create_input(0, graph_converter)

        if 'dim' in args and 'keepdim' in args and self.input_tensors[args['dim']] is not None:
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

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


class ATenFrobeniusNormOperator(ATenFrobeniusNormSchema):
    def parse_common(self, node, attrs, args, graph_converter):

        assert 'p' not in args
        self.input_tensors.insert(1, 2)
        ATenNormOperator.parse_common(self, node, attrs, args, graph_converter)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


class ATenLinalgVectorNormOperator(ATenLinalgVectorNormSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        ATenNormOperator.parse_common(self, node, attrs, args, graph_converter)


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
        assert input_tensor.tensor.ndim in (
            2,
            3,
        ), "Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported"
        output_tensors = self.to_tfl_tensors(self.output_names, self.output_tensors)

        output_size_h, output_size_w = self.input_tensors[1]
        kernel_h, kernel_w = self.input_tensors[2]
        dilation_h, dilation_w = self.input_tensors[3]
        padding_h, padding_w = self.input_tensors[4]
        stride_h, stride_w = self.input_tensors[5]

        fold_out = torch.nn.functional.fold(
            torch.from_numpy(input_tensor.tensor),
            (output_size_h, output_size_w),
            (kernel_h, kernel_w),
            (dilation_h, dilation_w),
            (padding_h, padding_w),
            (stride_h, stride_w),
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
        fake_output = fake_input[..., padding_h : output_size_h + padding_h, padding_w : output_size_w + padding_w].to(
            dtype=torch.int64
        )
        indices = torch.nonzero(fake_input >= 0)[fake_output].to(dtype=torch.int32)
        indices_tensor = self.create_attr_tensor(indices)
        graph_converter.add_operator(tfl.GatherNdOperator([padded_fold_out_tensor, indices_tensor], output_tensors))


class ATenAddbmmOperator(ATenAddbmmSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor, batch1_tensor, batch2_tensor = [self.find_or_create_input(i, graph_converter) for i in range(3)]
        output_tensors = self.to_tfl_tensors(self.output_names, self.output_tensors)
        assert (
            batch1_tensor.tensor.ndim == batch2_tensor.tensor.ndim == 3
        ), "batch1 and batch2 must be 3-D tensors each containing the same number of matrices"

        bmm_out = torch.bmm(torch.from_numpy(batch1_tensor.tensor), torch.from_numpy(batch2_tensor.tensor))
        bmm_out_tensor = self.create_transform_tensor(bmm_out)
        graph_converter.add_operator(tfl.BatchMatmulOperator([batch1_tensor, batch2_tensor], [bmm_out_tensor]))

        sum_bmm_out = torch.sum(bmm_out, dim=0)
        sum_bmm_out_tensor = self.create_transform_tensor(sum_bmm_out)
        dim_t = self.create_attr_tensor(np.array([0], dtype='int32'))
        graph_converter.add_operator(tfl.SumOperator([bmm_out_tensor, dim_t], [sum_bmm_out_tensor], keepDims=False))
        graph_converter.add_operator(tfl.AddOperator([input_tensor, sum_bmm_out_tensor], output_tensors))


class ATenBaddbmmOperator(ATenBaddbmmSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        input_tensor, batch1_tensor, batch2_tensor = [self.find_or_create_input(i, graph_converter) for i in range(3)]
        output_tensors = self.to_tfl_tensors(self.output_names, self.output_tensors)
        assert (
            batch1_tensor.tensor.ndim == batch2_tensor.tensor.ndim == 3
        ), "batch1 and batch2 must be 3-D tensors each containing the same number of matrices"

        bmm_out = torch.bmm(torch.from_numpy(batch1_tensor.tensor), torch.from_numpy(batch2_tensor.tensor))
        bmm_out_tensor = self.create_transform_tensor(bmm_out)
        graph_converter.add_operator(tfl.BatchMatmulOperator([batch1_tensor, batch2_tensor], [bmm_out_tensor]))
        graph_converter.add_operator(tfl.AddOperator([input_tensor, bmm_out_tensor], output_tensors))


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


class ATenBroadcastTensorsOperator(ATenBroadcastTensorsSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        input_names = graph_converter.get_list_expanded_names(self.input_names[0])
        inputs = self.to_tfl_tensors(
            input_names, self.input_tensors[0], graph_converter=graph_converter, non_existent_as_buffer=True
        )

        output_names = [f'{self.output_names[0]}:{i}' for i in range(len(input_names))]
        outputs = self.to_tfl_tensors(output_names, self.output_tensors[0])
        graph_converter.add_iterable_pair(self.output_names, output_names, 'input')

        ops = []
        for inp, outp in zip(inputs, outputs):
            input_shape = inp.shape
            output_shape = outp.shape

            # No-OP if input tensor is already of desired sizes
            if output_shape == input_shape:
                inputs = [inp, self.create_attr_tensor(inp.shape)]

                ops.append(tfl.ReshapeOperator(inputs, [outp], inp.shape))
                continue

            new_shape = input_shape
            actual_input = inp
            if len(output_shape) > len(input_shape):
                new_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
                new_shape_arr = np.array(new_shape, dtype='int32')
                new_shape_tensor = self.create_attr_tensor(new_shape_arr)
                reshaped = self.create_transform_tensor(np.reshape(inp.tensor, new_shape_arr))
                actual_input = reshaped
                reshape_op = tfl.ReshapeOperator([inp, new_shape_tensor], [reshaped], new_shape_arr)
                reshape_op.extra_hints['direction'] = 'up'
                ops.append(reshape_op)

            repeats = []
            for x, y in zip(new_shape, output_shape):
                if x != y:
                    repeats.append(y)
                else:
                    repeats.append(1)

            repeat_tensor = self.create_attr_tensor(np.array(repeats, dtype='int32'))
            ops.append(tfl.TileOperator([actual_input, repeat_tensor], [outp]))

        for op in ops:
            graph_converter.add_operator(op)
