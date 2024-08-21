import copy
import functools

import igraph as ig
import numpy as np
import torch

from tinynn.util.util import class_conditional, get_logger

from . import tflite as tfl
from .base import ExtendedOperator
from .graph import CommonGraph

log = get_logger(__name__)


WEIGHT_MAPPING = {
    ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: [1, 2, 3, 4, 5, 6, 7, 8],
    ExtendedOperator.BIDIRECTIONAL_SEQUENCE_LSTM: [1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25],
}

BIAS_MAPPING = {
    ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: {1: 12, 2: 13, 3: 14, 4: 15},
}

STATE_MAPPING = {
    ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: [18],
}

CELL_STATE_MAPPING = {
    ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: [19],
}


class HybridQuantizer(object):
    graph: CommonGraph

    def __init__(
        self, graph, asymmetric, q_type, per_channel, enable_conv, enable_int16_lstm, gen_single_op_models, config
    ) -> None:
        super().__init__()

        self.graph = graph
        self.asymmetric = asymmetric
        self.q_type = q_type
        self.per_channel = per_channel
        self.enable_conv = enable_conv
        self.enable_int16_lstm = enable_int16_lstm
        self.gen_single_op_models = gen_single_op_models

        if config is None:
            config = {}

        self.config = config

    def quantize(self):
        self.quantize_pass()
        self.int16_lstm_pass()

    @class_conditional(lambda self: self.enable_int16_lstm)
    def int16_lstm_pass(self):
        filtered_nodes = self.graph.graph.vs.select(functools.partial(is_int16_quantizable_lstm_node))

        actions = []
        replaced_tensors = {}
        for node in filtered_nodes:
            if self.config.get(node['outputs'][0], True) is False:
                continue

            if node['node_type'] == ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM:
                lstm_input = node['op'].inputs[0]
                if lstm_input.dtype == np.int8:
                    bias_indices = BIAS_MAPPING.get(node['node_type'])
                    for weight_idx, bias_idx in bias_indices.items():
                        bias_t = node['op'].inputs[bias_idx]
                        weight_t = node['op'].inputs[weight_idx]
                        name = bias_t.name
                        new_name = f'{name}_hybrid_q'
                        bias_a = np.frombuffer(bias_t.buffer.data, dtype='float32').reshape(bias_t.shape)
                        bias = torch.from_numpy(bias_a.copy())

                        bias_scale = weight_t.quantization.scale * lstm_input.quantization.scale
                        new_bias = torch.round(bias.detach() / bias_scale).to(dtype=torch.int32)
                        new_bias_t = tfl.Tensor(tfl.FakeQuantTensor(new_bias, bias_scale, 0), new_name)

                        replaced_tensors.setdefault(new_bias_t.name, new_bias_t)
                        new_bias_t = replaced_tensors[new_bias_t.name]
                        actions.append((self.graph.replace_operator_input, (node, bias_idx, new_bias_t)))

                    state_indices = STATE_MAPPING.get(node['node_type'])
                    for state_idx in state_indices:
                        node['op'].inputs[state_idx].quantization = copy.deepcopy(node['op'].outputs[0].quantization)
                        node['op'].inputs[state_idx].tensor = node['op'].inputs[state_idx].tensor.astype(np.int8)
                        node['op'].inputs[state_idx].dtype = node['op'].inputs[state_idx].tensor.dtype

                    cell_state_indices = CELL_STATE_MAPPING.get(node['node_type'])
                    for cell_state_idx in cell_state_indices:
                        q_cell_output = self.graph.rev_q_mapping[node['op'].extra_hints['cell_output']].quantization
                        q_cell_max = q_cell_output.scale * (127 - q_cell_output.zero_point)
                        q_cell_min = q_cell_output.scale * (-128 - q_cell_output.zero_point)
                        q_cell_abs_max = np.maximum(np.abs(q_cell_max), np.abs(q_cell_min))
                        cell_pot = np.power(2, np.maximum(np.ceil(np.log2(q_cell_abs_max)), 0)).item()
                        node['op'].inputs[cell_state_idx].quantization = tfl.QuantizationParameters(cell_pot / 32768, 0)
                        node['op'].inputs[cell_state_idx].tensor = (
                            node['op'].inputs[cell_state_idx].tensor.astype(np.int16)
                        )
                        node['op'].inputs[cell_state_idx].dtype = node['op'].inputs[cell_state_idx].tensor.dtype

                    # Add intermediates for int8x8_16 lstm
                    name = node['op'].outputs[0].name
                    input_to_input_intermediate = tfl.Tensor(np.zeros(0, dtype='float32'), f'{name}_intermediate_1')
                    input_to_forget_intermediate = tfl.Tensor(np.zeros(0, dtype='float32'), f'{name}_intermediate_2')
                    input_to_cell_intermediate = tfl.Tensor(np.zeros(0, dtype='float32'), f'{name}_intermediate_3')
                    input_to_output_intermediate = tfl.Tensor(np.zeros(0, dtype='float32'), f'{name}_intermediate_4')
                    effective_hidden_scale_intermediate = tfl.Tensor(
                        tfl.FakeQuantTensor(np.zeros(0, dtype='int8'), node['op'].outputs[0].quantization.scale, 0),
                        f'{name}_intermediate_5',
                    )

                    actions.append((self.graph.append_operator_input, (node, input_to_input_intermediate, True)))
                    actions.append((self.graph.append_operator_input, (node, input_to_forget_intermediate, True)))
                    actions.append((self.graph.append_operator_input, (node, input_to_cell_intermediate, True)))
                    actions.append((self.graph.append_operator_input, (node, input_to_output_intermediate, True)))
                    actions.append(
                        (self.graph.append_operator_input, (node, effective_hidden_scale_intermediate, True))
                    )

        for func, args in actions:
            func(*args)

    def quantize_pass(self):
        filtered_nodes = self.graph.graph.vs.select(functools.partial(is_quantizable_node, with_conv=self.enable_conv))

        actions = []
        replaced_tensors = {}
        for node in filtered_nodes:
            if self.config.get(node['outputs'][0], True) is False:
                continue
            weight_indices = WEIGHT_MAPPING.get(node['node_type'], [1])
            skip = False
            for weight_idx in weight_indices:
                new_weight = None
                weight_t = node['op'].inputs[weight_idx]
                if weight_t.buffer is None or str(weight_t.dtype) != 'float32':
                    skip = True
                    break
            if skip:
                continue
            for weight_idx in weight_indices:
                weight_t = node['op'].inputs[weight_idx]
                name = weight_t.name
                weight_a = np.frombuffer(weight_t.buffer.data, dtype='float32').reshape(weight_t.shape)
                weight = torch.from_numpy(weight_a.copy())
                if (
                    node['node_type']
                    in (
                        ExtendedOperator.FULLY_CONNECTED,
                        ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
                        ExtendedOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
                    )
                    or not self.per_channel
                ):
                    if node['node_type'] == ExtendedOperator.DEPTHWISE_CONV_2D:
                        log.warning('DEPTHWISE_CONV_2D doesn\'t support hybrid per-tensor quantization')
                        continue
                    if self.asymmetric and hasattr(node['op'], 'asymmetricQuantizeInputs'):
                        node['op'].asymmetricQuantizeInputs = True
                    if self.q_type == np.uint8:
                        new_weight = quantize(name, weight, torch.qint8, torch.per_tensor_symmetric, q_type=np.int8)
                        new_weight.reinterpret_as(self.q_type)
                    else:
                        new_weight = quantize(name, weight, torch.qint8, torch.per_tensor_symmetric, q_type=self.q_type)
                elif node['node_type'] == ExtendedOperator.CONV_2D:
                    new_weight = quantize(name, weight, torch.qint8, torch.per_channel_symmetric, 0, q_type=self.q_type)
                elif node['node_type'] == ExtendedOperator.DEPTHWISE_CONV_2D:
                    new_weight = quantize(
                        name, weight, torch.qint8, torch.per_channel_symmetric, -1, q_type=self.q_type
                    )

                if self.gen_single_op_models:
                    node['op'].extra_hints['orig_float'] = copy.deepcopy(node['op'])

                replaced_tensors.setdefault(new_weight.name, new_weight)
                new_weight = replaced_tensors[new_weight.name]
                actions.append((self.graph.replace_operator_input, (node, weight_idx, new_weight)))

        for func, args in actions:
            func(*args)


def is_quantizable_node(vertex: ig.Vertex, with_conv: bool):
    return vertex['node_type'] in (
        ExtendedOperator.FULLY_CONNECTED,
        ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
        ExtendedOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
    ) or (
        with_conv
        and vertex['node_type']
        in (
            ExtendedOperator.CONV_2D,
            ExtendedOperator.DEPTHWISE_CONV_2D,
        )
    )


def is_int16_quantizable_lstm_node(vertex: ig.Vertex):
    return vertex['node_type'] in (ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,)


def quantize(name, tensor, dtype, qscheme, axis=None, q_type=np.uint8):
    assert qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric)

    new_name = f'{name}_hybrid_q'

    if dtype == torch.quint8:
        quant_min, quant_max = 0, 255
    else:
        quant_min, quant_max = -127, 127

    if axis is not None:
        if axis < 0:
            axis += tensor.ndim
        dim = [i for i in range(tensor.ndim) if i != axis]
    else:
        dim = None

    if hasattr(torch, 'amin') and hasattr(torch, 'amax'):
        min_val = torch.amin(tensor, dim)
        max_val = torch.amax(tensor, dim)
    else:
        if dim is None:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
        else:
            orig_dim = tensor.size(axis)
            if axis != 0:
                perm = [axis] + dim
                tensor_perm = tensor.permute(perm)
            else:
                tensor_perm = tensor
            tensor_2d = tensor_perm.reshape(orig_dim, -1)
            min_val, _ = torch.min(tensor_2d, 1)
            max_val, _ = torch.max(tensor_2d, 1)

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64)

    eps = torch.tensor(torch.finfo(torch.float32).eps)

    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    scale = torch.max(scale, eps)
    if dtype == torch.quint8:
        zero_point = zero_point.new_full(zero_point.size(), 128)

    if qscheme == torch.per_channel_symmetric:
        q_tensor = torch.quantize_per_channel(tensor, scale, zero_point, axis, dtype)
    else:
        q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)

    return tfl.Tensor(q_tensor, new_name, q_type=q_type)
