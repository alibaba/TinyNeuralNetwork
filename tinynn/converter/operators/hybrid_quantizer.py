import functools
import igraph as ig
import numpy as np

import torch

from . import tflite as tfl
from .base import ExtendedOperator
from .graph import CommonGraph

from tinynn.util.util import get_logger

log = get_logger(__name__)


WEIGHT_MAPPING = {
    ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: [1, 2, 3, 4, 5, 6, 7, 8],
    ExtendedOperator.BIDIRECTIONAL_SEQUENCE_LSTM: [1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25],
}


class HybridQuantizer(object):
    graph: CommonGraph

    def __init__(self, graph, asymmetric, q_type, per_channel, enable_conv) -> None:
        super().__init__()

        self.graph = graph
        self.asymmetric = asymmetric
        self.q_type = q_type
        self.per_channel = per_channel
        self.enable_conv = enable_conv

    def quantize(self):
        self.quantize_pass()

    def quantize_pass(self):
        filtered_nodes = self.graph.graph.vs.select(functools.partial(is_quantizable_node, with_conv=self.enable_conv))

        actions = []
        for node in filtered_nodes:
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
