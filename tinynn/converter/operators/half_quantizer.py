import functools
import typing

import igraph as ig
import numpy as np

import torch

from . import tflite as tfl
from .base import ExtendedOperator
from .graph import CommonGraph

from tinynn.util.util import get_logger

log = get_logger(__name__)


class HalfQuantizer(object):
    graph: CommonGraph

    def __init__(self, graph) -> None:
        super().__init__()

        self.graph = graph
        self.fuse_tensor_count = 0
        self.fuse_attr_count = 0

    def create_attr_tensor(
        self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None
    ):
        if name is None:
            if self.fuse_attr_count == 0:
                name = 'half_attr'
            else:
                name = f'half_attr_{self.fuse_attr_count}'
            self.fuse_attr_count += 1
        return tfl.Tensor(tensor, name, has_buffer=True, quantization=quantization)

    def create_transform_tensor(
        self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None
    ):
        if name is None:
            if self.fuse_tensor_count == 0:
                name = 'half_transform'
            else:
                name = f'half_transform_{self.fuse_tensor_count}'
            self.fuse_tensor_count += 1
        return tfl.Tensor(tensor, name, has_buffer=False, quantization=quantization)

    def quantize(self):
        self.quantize_pass()

    def quantize_pass(self):
        filtered_nodes = self.graph.graph.vs.select(functools.partial(is_quantizable_node, graph_converter=self.graph))
        actions = []
        for node in filtered_nodes:
            tn = node['name']
            t = self.graph.tensor_map[tn]
            c = self.create_attr_tensor(t.tensor.astype('float16'))
            new_t = self.create_transform_tensor(t.tensor)
            op = tfl.DequantizeOperator([c], [new_t])
            self.graph.add_operator(op)

            next_ops = set()
            node_map = {}
            for out_edge in node.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                next_op = next_node['op']
                if next_op not in next_ops:
                    next_ops.add(next_op)
                    node_map[next_op] = next_node

            for next_op in next_ops:
                next_node = node_map[next_op]
                for i, inp in enumerate(next_op.inputs):
                    if inp.name == tn:
                        actions.append((self.graph.replace_operator_input, (next_node, i, new_t)))

        for func, args in actions:
            func(*args)


def is_quantizable_node(vertex: ig.Vertex, graph_converter: CommonGraph):
    return (
        vertex['node_type'] == ExtendedOperator.CONSTANT_NODE
        and str(graph_converter.tensor_map[vertex['name']].dtype) == 'float32'
    )
