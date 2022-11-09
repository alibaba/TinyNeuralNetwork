import copy
import functools
import itertools
import re
import typing
import warnings

import igraph as ig
import numpy as np

from tinynn.util.util import class_conditional, get_logger

from ..schemas.tflite.schema_generated import ActivationFunctionType, Padding
from . import tflite as tfl
from .base import FUSE_ACTIVATION_MAP, ExtendedOperator
from .graph import CommonGraph

log = get_logger(__name__, 'INFO')


class GraphOptimizer(object):
    graph: CommonGraph
    fuse_tensor_count: int
    fuse_attr_count: int
    fuse_quant: bool
    group_conv_rewrite: bool
    tflite_micro_rewrite: bool
    quantize_input_output_type: typing.Optional[str]

    # Optimization levels
    NO_OPTIMIZE: int = 0
    FOLD_BUFFER: int = 1
    FUSE_BN: int = 2
    COMMON_OPTIMIZE: int = 3
    BRANCH_OPTIMIZE: int = 4
    BRANCH_OPTIMIZE_EXTENDED: int = 5
    ALL_OPTIMIZE: int = 5

    def __init__(
        self,
        graph: CommonGraph,
        level: int,
        fuse_quant: bool,
        group_conv_rewrite: bool,
        rewrite_quantizable: bool,
        tflite_micro_rewrite: bool,
        quantize_input_output_type: typing.Optional[str],
        fuse_input_indices: typing.Optional[typing.List[int]] = None,
        fuse_output_indices: typing.Optional[typing.List[int]] = None,
        max_transpose_dims: int = -1,
    ) -> None:
        self.graph = graph
        self.fuse_tensor_count = 0
        self.fuse_attr_count = 0
        self.level = level
        self.fuse_quant = fuse_quant
        self.group_conv_rewrite = group_conv_rewrite
        self.rewrite_quantizable = rewrite_quantizable
        self.tflite_micro_rewrite = tflite_micro_rewrite
        self.quantize_input_output_type = quantize_input_output_type
        self.fuse_input_indices = fuse_input_indices
        self.fuse_output_indices = fuse_output_indices
        self.max_transpose_dims = max_transpose_dims

    def create_attr_tensor(
        self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None
    ):
        if name is None:
            if self.fuse_attr_count == 0:
                name = 'fuse_attr'
            else:
                name = f'fuse_attr_{self.fuse_attr_count}'
            self.fuse_attr_count += 1
        return tfl.Tensor(tensor, name, has_buffer=True, quantization=quantization)

    def create_transform_tensor(
        self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None
    ):
        if name is None:
            if self.fuse_tensor_count == 0:
                name = 'fuse_transform'
            else:
                name = f'fuse_transform_{self.fuse_tensor_count}'
            self.fuse_tensor_count += 1
        return tfl.Tensor(tensor, name, has_buffer=False, quantization=quantization)

    @class_conditional(lambda self: self.level >= GraphOptimizer.FUSE_BN)
    def fuse_conv_fc_bn(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(functools.partial(is_bn_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        actions = []
        for conv, bn, tensor in filtered_pairs:
            bn_activ = bn['op'].fusedActivationFunction
            conv_activ = getattr(conv['op'], 'fusedActivationFunction', None)
            if conv_activ is None and bn_activ != ActivationFunctionType.NONE:
                continue

            # Find out the output of the batch-norm nodes
            new_output = bn['outputs'][0]
            assert new_output in self.graph.tensor_map

            # For each node that is next of a batch-norm node, we connect it with the conv node
            self.graph.connect_next_tensors(bn, conv, new_output)

            # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the
            # batch-norm instead
            conv['outputs'][0] = new_output
            conv['op'].outputs[0] = self.graph.tensor_map[new_output]
            self.graph.tensor_node_map[new_output] = conv['name']
            tensor['name'] = bn['outputs'][0]
            tensor['label'] = bn['outputs'][0]

            if bn_activ != ActivationFunctionType.NONE and conv_activ == ActivationFunctionType.NONE:
                conv['op'].fusedActivationFunction = bn_activ

            # Collect the arguments of the conv and batch-norm nodes
            weight = conv['op'].inputs[1]
            bias = conv['op'].inputs[2] if len(conv['op'].inputs) > 2 else None
            bn_w, bn_b, bn_mean, bn_var = bn['op'].inputs[1:]
            bn_w, bn_b, bn_mean, bn_var = (
                bn_w.tensor.copy(),
                bn_b.tensor.copy(),
                bn_mean.tensor.copy(),
                bn_var.tensor.copy(),
            )
            activ_w = weight.tensor.copy()
            activ_b = bias.tensor.copy() if bias is not None else None
            eps = bn['op'].eps

            # Fuse conv/fc and batch-norm
            new_weight = fuse_bn_weight(
                eps, bn_w, bn_var, activ_w, conv['node_type'] == ExtendedOperator.GENERIC_DECONV
            )
            new_bias = fuse_bn_bias(eps, bn_w, bn_var, bn_mean, bn_b, activ_b)

            # New attribute tensors
            new_w = self.create_attr_tensor(new_weight)
            new_b = self.create_attr_tensor(new_bias)

            # Collect the actions we should take here
            # The reason that we don't do the actions here is because we are currently in the loop of vertices,
            # the iterator will be invalidated once `replace_operator_input` is called
            actions.append((self.graph.replace_operator_input, (conv, 1, new_w)))
            if bias is not None:
                actions.append((self.graph.replace_operator_input, (conv, 2, new_b)))
            else:
                actions.append((self.graph.append_operator_input, (conv, new_b)))

            remove_ids.append(bn.index)

        # Process actions
        for func, args in actions:
            func(*args)

        # Delete batch-norm nodes
        for id in remove_ids:
            vertex = self.graph.graph.vs[id]
            assert vertex['node_type'] == ExtendedOperator.BATCH_NORM
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_activation(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(functools.partial(is_activ_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for pre_activ, activ, tensor in filtered_pairs:
            # Find out the output of the batch-norm nodes
            new_output = activ['outputs'][0]
            assert new_output in self.graph.tensor_map

            # For each node that is next of the activation node, we connect it with the previous node
            self.graph.connect_next_tensors(activ, pre_activ, new_output)

            # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the
            # batch-norm instead
            pre_activ['outputs'][0] = new_output
            pre_activ['op'].outputs[0] = self.graph.tensor_map[new_output]
            self.graph.tensor_node_map[new_output] = pre_activ['name']
            tensor['name'] = activ['outputs'][0]
            tensor['label'] = activ['outputs'][0]

            # Fuse activation
            pre_activ['op'].fusedActivationFunction = FUSE_ACTIVATION_MAP[activ['node_type']]

            remove_ids.append(activ.index)

        # Delete activation nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_same_padding(self):
        edges = self.graph.graph.es.select(functools.partial(is_padding_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]) for x in edges)

        def _remove_last_pred(seq):
            op = seq[1]['op']
            return False, op

        def _remove_last_action(first_node, last_node, custom_data):
            op = custom_data
            op.padding = Padding.SAME
            return []

        def _skip_pred(seq):
            pad_op = seq[0]['op']
            next_op = seq[1]['op']

            input_shape = pad_op.inputs[0].shape[1:-1]
            if seq[1]['node_type'] == ExtendedOperator.MAX_POOL_2D:
                kernel_shape = (next_op.filterHeight, next_op.filterWidth)
                strides = (next_op.strideH, next_op.strideW)
                dilation = (1, 1)
            elif seq[1]['node_type'] in (
                ExtendedOperator.CONV_2D,
                ExtendedOperator.DEPTHWISE_CONV_2D,
            ):
                kernel_shape = next_op.inputs[1].shape[1:-1]
                strides = (next_op.strideH, next_op.strideW)
                dilation = (next_op.dilationHFactor, next_op.dilationWFactor)
            elif seq[1]['node_type'] == ExtendedOperator.CONV_3D:
                kernel_shape = next_op.inputs[1].shape[:3]
                strides = (next_op.strideD, next_op.strideH, next_op.strideW)
                dilation = (next_op.dilationDFactor, next_op.dilationHFactor, next_op.dilationWFactor)

            pad_args = get_same_padding_args(input_shape, kernel_shape, strides, dilation)
            pad_arr = np.array(pad_args, dtype='int32')

            old_pad_arr = pad_op.inputs[1].tensor
            skip = not np.array_equal(pad_arr, old_pad_arr)

            return skip

        elinimate_sequences(
            self.graph,
            filtered_pairs,
            True,
            None,
            _remove_last_pred,
            _remove_last_action,
            _skip_pred,
            force_forward_input=True,
        )

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_same_padding_slicing(self):
        edges = self.graph.graph.es.select(functools.partial(is_slicing_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        actions = []
        for prev_node, slice_node, tensor in filtered_pairs:
            prev_op = prev_node['op']
            slice_op = slice_node['op']

            input_shape = slice_op.outputs[0].shape[1:-1]
            if prev_node['node_type'] == ExtendedOperator.TRANSPOSE_CONV:
                kernel_shape = prev_op.inputs[1].shape[1:-1]
                strides = (prev_op.strideH, prev_op.strideW)
                dilation = (1, 1)
            elif prev_node['node_type'] == ExtendedOperator.CONV_3D_TRANSPOSE:
                kernel_shape = prev_op.inputs[1].shape[:3]
                strides = (prev_op.strideD, prev_op.strideH, prev_op.strideW)
                dilation = (prev_op.dilationDFactor, prev_op.dilationHFactor, prev_op.dilationWFactor)

            pad_args = get_same_padding_args(input_shape, kernel_shape, strides, dilation)
            pad_arr = np.array(pad_args, dtype='int32')

            start_arr = [x for x in slice_op.inputs[1].tensor]
            end_arr = [slice_op.inputs[0].shape[i] - x - slice_op.outputs[0].shape[i] for i, x in enumerate(start_arr)]

            old_pad_args = [[x, y] for x, y in zip(start_arr, end_arr)]
            skip = not np.array_equal(pad_arr, old_pad_args)

            if skip:
                continue

            # Find out the output of the slice nodes
            new_output = slice_node['outputs'][0]
            assert new_output in self.graph.tensor_map

            # For each node that is next of the slice_nodeation node, we connect it with the previous node
            self.graph.connect_next_tensors(slice_node, prev_node, new_output)

            # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the
            # slice op instead
            prev_node['outputs'][0] = new_output
            prev_node['op'].outputs[0] = self.graph.tensor_map[new_output]
            self.graph.tensor_node_map[new_output] = prev_node['name']
            tensor['name'] = slice_node['outputs'][0]
            tensor['label'] = slice_node['outputs'][0]

            # Fuse padding
            prev_node['op'].padding = Padding.SAME

            new_shape = np.array(prev_node['op'].outputs[0].shape, dtype='int32')
            new_shape_tensor = self.create_attr_tensor(new_shape)
            actions.append((self.graph.replace_operator_input, (prev_node, 0, new_shape_tensor)))

            remove_ids.append(slice_node.index)

        for func, args in actions:
            func(*args)

        # Delete activation nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_requantize(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(
            functools.partial(is_requantize_fusable_edge, graph_converter=self.graph.graph)
        )
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for pre_activ, activ, tensor in filtered_pairs:
            if pre_activ.outdegree() > 1:
                skip = False
                pre_quantize = None
                for out_edge in pre_activ.out_edges():
                    next_node = self.graph.graph.vs[out_edge.target]
                    while True:
                        if next_node['node_type'] == ExtendedOperator.QUANTIZE:
                            if pre_quantize is None:
                                pre_quantize = next_node['op'].outputs[0].quantization
                            else:
                                cur_quantize = next_node['op'].outputs[0].quantization
                                if (
                                    pre_quantize.scale != cur_quantize.scale
                                    or pre_quantize.zero_point != cur_quantize.zero_point
                                    or pre_quantize.dim != cur_quantize.dim
                                ):
                                    skip = True
                            break
                        elif next_node['node_type'] == ExtendedOperator.DEQUANTIZE:
                            break
                        elif next_node['node_type'] in (ExtendedOperator.RESHAPE, ExtendedOperator.TRANSPOSE):
                            if next_node.outdegree() > 1:
                                skip = True
                                break
                            else:
                                next_node = self.graph.graph.vs[next_node.out_edges()[0].target]
                        else:
                            skip = True
                            break

                    if skip:
                        break

                if skip:
                    continue

                # Find out the output of the first node in the sequence
                output_name = activ['op'].inputs[0].name
                output_idx = pre_activ['outputs'].index(output_name)
                new_output = pre_activ['outputs'][output_idx]
                assert new_output in self.graph.tensor_map

                # For each node that is next of the last node, we connect it with the first node
                # Also, the replace the tensors when needed
                self.graph.replace_next_tensors(activ, pre_activ, new_output)

                new_tensor = pre_activ['op'].outputs[0]
                old_tensor = activ['op'].outputs[0]
                new_tensor.quantization = old_tensor.quantization
            else:
                # Find out the output of the batch-norm nodes
                new_output = activ['outputs'][0]
                assert new_output in self.graph.tensor_map

                # For each node that is next of the activation node, we connect it with the previous node
                self.graph.connect_next_tensors(activ, pre_activ, new_output)

                # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the
                # batch-norm instead
                pre_activ['outputs'][0] = new_output
                pre_activ['op'].outputs[0] = self.graph.tensor_map[new_output]
                self.graph.tensor_node_map[new_output] = pre_activ['name']
                tensor['name'] = activ['outputs'][0]
                tensor['label'] = activ['outputs'][0]

            remove_ids.append(activ.index)

        # Delete activation nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_reciprocal_sqrt(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(functools.partial(is_reciprocal_sqrt_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for sqrt, div, tensor in filtered_pairs:
            sqrt['node_type'] = ExtendedOperator.RSQRT
            sqrt['op'] = tfl.RsqrtOperator(sqrt['op'].inputs, sqrt['op'].outputs)

            div_op = div['op']
            if (
                div_op.inputs[0].buffer is not None
                and np.all(div_op.inputs[0].tensor == 1.0)
                and div['op'].fusedActivationFunction == ActivationFunctionType.NONE
            ):
                new_output = div['outputs'][0]
                assert new_output in self.graph.tensor_map

                # For each node that is next of the div node, we connect it with the previous node
                self.graph.connect_next_tensors(div, sqrt, new_output)

                # Update graph, prepare to drop the output tensor of the div node and use the output tensor of the
                # sqrt instead
                sqrt['outputs'][0] = new_output
                sqrt['op'].outputs[0] = self.graph.tensor_map[new_output]
                self.graph.tensor_node_map[new_output] = sqrt['name']
                tensor['name'] = div['outputs'][0]
                tensor['label'] = div['outputs'][0]

                # remove div op
                remove_ids.append(div.index)
            else:
                div['node_type'] = ExtendedOperator.MUL
                div['op'] = tfl.MulOperator(
                    div['op'].inputs, div['op'].outputs, fusedActivationFunction=div['op'].fusedActivationFunction
                )

        # Delete div nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_conv2d_gather(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(functools.partial(is_conv2d_gather_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        actions = []
        for conv, gather, tensor in filtered_pairs:
            # Find out the output of the batch-norm nodes
            new_output = gather['outputs'][0]
            assert new_output in self.graph.tensor_map

            # For each node that is next of the activation node, we connect it with the previous node
            self.graph.connect_next_tensors(gather, conv, new_output)

            # Update graph, prepare to drop the output tensor of the gather node and use the output tensor of the
            # conv instead
            conv['outputs'][0] = new_output
            conv_out_quant_param = conv['op'].outputs[0].quantization
            conv['op'].outputs[0] = self.graph.tensor_map[new_output]
            conv['op'].outputs[0].quantization = conv_out_quant_param
            self.graph.tensor_node_map[new_output] = conv['name']
            tensor['name'] = gather['outputs'][0]
            tensor['label'] = gather['outputs'][0]
            # permute weight of conv-op
            indx = gather['op'].inputs[1].tensor.copy()
            w = conv['op'].inputs[1].tensor.copy()
            w_quant_param = conv['op'].inputs[1].quantization
            new_w = np.take(w, indx, axis=0)
            # permute bias of conv-op
            b = conv['op'].inputs[2].tensor.copy() if len(conv['op'].inputs) > 2 else None
            b_quant_param = conv['op'].inputs[2].quantization
            new_b = np.take(b, indx, axis=0) if b is not None else None
            if w_quant_param is not None and isinstance(w_quant_param.scale, list) and w_quant_param.dim == 0:
                new_w_scale = np.take(w_quant_param.scale, indx, axis=0)
                new_w_zeros = np.take(w_quant_param.zero_point, indx, axis=0)
                w_quant_param.scale = new_w_scale
                w_quant_param.zero_point = new_w_zeros
                if new_b is not None:
                    new_b_scale = np.take(b_quant_param.scale, indx, axis=0)
                    new_b_zeros = np.take(b_quant_param.zero_point, indx, axis=0)
                    b_quant_param.scale = new_b_scale
                    b_quant_param.zero_point = new_b_zeros
            new_w = self.create_attr_tensor(new_w, quantization=w_quant_param)
            actions.append((self.graph.replace_operator_input, (conv, 1, new_w)))
            new_b = self.create_attr_tensor(new_b, quantization=b_quant_param)
            actions.append((self.graph.replace_operator_input, (conv, 2, new_b)))

            # remove gather op
            remove_ids.append(gather.index)

        # Process actions
        for func, args in actions:
            func(*args)
        # Delete activation nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.tflite_micro_rewrite)
    def split_requantize(self):
        vertices = self.graph.graph.vs.select(functools.partial(is_requantize_node, graph_converter=self.graph.graph))

        remove_ids = []
        ops = []
        restore_mapping = []
        for quantize in vertices:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in quantize.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in quantize['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            remove_ids.append(quantize.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(vertices, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for quantize, mapping in zip(sorted_ops, restore_mapping):
            input_tensor = quantize.inputs[0]
            output_tensor = quantize.outputs[0]

            intermediate = self.create_transform_tensor(input_tensor.tensor.astype('float32'))

            ops.append(tfl.DequantizeOperator([input_tensor], [intermediate]))
            ops.append(tfl.QuantizeOperator([intermediate], [output_tensor]))

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    def transform_graph(self):
        # Find transformable ops
        filtered_nodes = self.graph.graph.vs.select(
            functools.partial(is_transformable_node, graph_converter=self.graph.graph)
        )
        remove_ids = []
        ops = []
        restore_mapping = []
        for node in filtered_nodes:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in node.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in node['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            ops.append(node)
            remove_ids.append(node.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(ops, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        # Do transformation
        for op, mapping in zip(sorted_ops, restore_mapping):
            op.transform(self.graph, mapping)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_simple_transpose_pass(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_transpose_fusable_edge, graph_converter=self.graph.graph)
        )
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            new_perm = fuse_transpose_perms(seq)

            hints = set()
            for node in seq:
                if 'direction' in node['op'].extra_hints:
                    hints.add(node['op'].extra_hints['direction'])

            if len(hints) == 1:
                hint = next(iter(hints))
            else:
                hint = None

            remove_first = np.array_equal(new_perm, np.sort(new_perm))
            return remove_first, (new_perm, hint)

        def _remove_first_action(first_node, last_node, custom_data):
            # Set fused perm to the first transpose node
            new_perm, hint = custom_data
            if hint is None:
                if 'direction' in first_node['op'].extra_hints:
                    del first_node['op'].extra_hints['direction']
            else:
                first_node['op'].extra_hints['direction'] = hint
            new_perm_tensor = self.create_attr_tensor(new_perm)
            action = (self.graph.replace_operator_input, (first_node, 1, new_perm_tensor))
            return [action]

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_dequant_quant_pass(self, q_first):
        edges = self.graph.graph.es.select(
            functools.partial(is_dequant_quant_fusable_edge, graph_converter=self.graph.graph, q_first=q_first)
        )
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        r_edges = self.graph.graph.es.select(
            functools.partial(is_dequant_quant_fusable_edge, graph_converter=self.graph.graph, q_first=not q_first)
        )
        r_filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in r_edges]

        filtered_pairs = fuse_connected_edges(filtered_pairs + r_filtered_pairs)

        new_pairs = []
        for seq in filtered_pairs:
            start_idx = 0
            end_idx = len(seq)

            if q_first:
                if seq[0]['node_type'] != ExtendedOperator.QUANTIZE:
                    start_idx += 1
                if seq[-1]['node_type'] != ExtendedOperator.DEQUANTIZE:
                    end_idx -= 1
            else:
                if seq[0]['node_type'] != ExtendedOperator.DEQUANTIZE:
                    start_idx += 1
                if seq[-1]['node_type'] != ExtendedOperator.QUANTIZE:
                    end_idx -= 1

            new_seq = seq[start_idx:end_idx]
            if len(new_seq) >= 2:
                new_pairs.append(new_seq)

        filtered_pairs = new_pairs

        def _remove_first_pred(seq):
            first_node, last_node = seq[0], seq[-1]
            new_qparams = last_node['op'].outputs[0].quantization
            orig_qparams = first_node['op'].inputs[0].quantization

            if (
                first_node['node_type'] == ExtendedOperator.DEQUANTIZE
                and last_node['node_type'] == ExtendedOperator.QUANTIZE
            ):
                assert new_qparams is not None
                assert orig_qparams is not None

                remove_first = (
                    new_qparams.scale == orig_qparams.scale
                    and new_qparams.zero_point == orig_qparams.zero_point
                    and new_qparams.dim == orig_qparams.dim
                )
            else:
                assert new_qparams is None
                assert orig_qparams is None

                remove_first = True

            return remove_first, None

        def _remove_first_action(first_node, last_node, custom_data):
            # Set new node type to first node
            first_node['node_type'] = ExtendedOperator.QUANTIZE
            old_op = first_node['op']
            first_node['op'] = tfl.QuantizeOperator(old_op.inputs, old_op.outputs)
            return []

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_simple_reshape_pass(self):
        edges = self.graph.graph.es.select(functools.partial(is_reshape_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edge
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            first_node, last_node = seq[0], seq[-1]
            new_shape = last_node['op'].inputs[1].tensor
            orig_shape = np.array(first_node['op'].inputs[0].shape, dtype='int32')

            hints = set()
            for node in seq:
                if 'direction' in node['op'].extra_hints:
                    hints.add(node['op'].extra_hints['direction'])

            if len(hints) == 1:
                hint = next(iter(hints))
            else:
                hint = None

            remove_first = np.array_equal(new_shape, orig_shape)
            return remove_first, (new_shape, hint)

        def _remove_first_action(first_node, last_node, custom_data):
            # Set final shape to the first reshape node
            new_shape, hint = custom_data
            if hint is None:
                if 'direction' in first_node['op'].extra_hints:
                    del first_node['op'].extra_hints['direction']
            else:
                first_node['op'].extra_hints['direction'] = hint
            new_shape_tensor = self.create_attr_tensor(np.array(new_shape, dtype='int32'))
            first_node['op'].newShape = new_shape_tensor.tensor
            action = (self.graph.replace_operator_input, (first_node, 1, new_shape_tensor))
            return [action]

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_simple_slice_pass(self):
        edges = self.graph.graph.es.select(functools.partial(is_slice_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edge
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            fused_info = fuse_slices(seq)

            return False, fused_info

        def _remove_first_action(first_node, last_node, custom_data):
            # Set final shape to the first reshape node
            start, end, stride = custom_data
            if all((x == 1 for x in stride)):
                target_class = tfl.SliceOperator
                target_type = ExtendedOperator.SLICE
            else:
                target_class = tfl.StridedSliceOperator
                target_type = ExtendedOperator.STRIDED_SLICE

            if target_type == ExtendedOperator.SLICE:
                size = end - start
                start_tensor = self.create_attr_tensor(np.array(start, dtype='int32'))
                size_tensor = self.create_attr_tensor(np.array(size, dtype='int32'))
                actions = [
                    (self.graph.replace_operator_input, (first_node, 1, start_tensor)),
                    (self.graph.replace_operator_input, (first_node, 2, size_tensor)),
                ]
                if first_node['node_type'] != ExtendedOperator.SLICE:
                    old_slice_op = first_node['op']
                    first_node['node_type'] = ExtendedOperator.SLICE
                    first_node['op'] = target_class(old_slice_op.inputs, old_slice_op.outputs)
                    actions.append((self.graph.remove_operator_input, (first_node, 3)))
            else:
                size = end - start
                start_tensor = self.create_attr_tensor(np.array(start, dtype='int32'))
                end_tensor = self.create_attr_tensor(np.array(end, dtype='int32'))
                stride_tensor = self.create_attr_tensor(np.array(stride, dtype='int32'))
                if first_node['node_type'] == ExtendedOperator.STRIDED_SLICE:
                    actions = [
                        (self.graph.replace_operator_input, (first_node, 1, start_tensor)),
                        (self.graph.replace_operator_input, (first_node, 2, end_tensor)),
                        (self.graph.replace_operator_input, (first_node, 3, stride_tensor)),
                    ]
                else:
                    old_slice_op = first_node['op']
                    first_node['node_type'] = ExtendedOperator.STRIDED_SLICE
                    first_node['op'] = target_class(old_slice_op.inputs, old_slice_op.outputs)
                    actions = [
                        (self.graph.replace_operator_input, (first_node, 1, start_tensor)),
                        (self.graph.replace_operator_input, (first_node, 2, end_tensor)),
                        (self.graph.append_operator_input, (first_node, stride_tensor)),
                    ]

            return actions

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    def cleanup_dead_nodes(self):
        cleanup_nodes = []
        if not self.graph.graph.is_connected('weak'):
            while True:
                for vertex in self.graph.graph.vs:
                    if (
                        vertex['node_type'] not in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE)
                        and vertex.outdegree() == 0
                    ):
                        if vertex['node_type'] == ExtendedOperator.INPUT_NODE:
                            continue
                        if vertex['node_type'] != ExtendedOperator.CONSTANT_NODE:
                            warnings.warn('Non constant node removed, something must be wrong there')
                            log.warning('-' * 30)
                            log.warning('Info of the deleted node:')
                            log.warning(f'vertex: {vertex}')
                            # edge = self.graph.graph.es.select(name=vertex['outputs'][0])
                            # assert edge is None, (
                            #     f'The edge {vertex["outputs"][0]} exists but the connection to the vertex'
                            #     f' {vertex["name"]} is broken, probably there have some conflicts in the names of the'
                            #     ' nodes'
                            # )
                        cleanup_nodes.append(vertex.index)

                if len(cleanup_nodes) == 0:
                    break

                self.graph.graph.delete_vertices(cleanup_nodes)
                cleanup_nodes.clear()

    @class_conditional(lambda self: self.level >= GraphOptimizer.FOLD_BUFFER)
    def fold_transpose_buffer(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_constant_transpose_fusable_edge, graph_converter=self.graph.graph)
        )
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for constant, transpose, tensor in filtered_pairs:
            # Calculate the output of the transposed constant nodes
            constant_tensor = transpose['op'].inputs[0].tensor
            perm_tensor = transpose['op'].inputs[1].tensor
            new_constant = np.transpose(constant_tensor, perm_tensor)
            new_tensor = self.create_attr_tensor(new_constant, quantization=transpose['op'].outputs[0].quantization)
            new_node = self.graph.add_nodes([new_tensor])[0]

            # For each node that is next of a constant transpose node, we connect it with the new constant node
            for out_edge in transpose.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                self.graph.graph.add_edge(new_node, next_node, name=new_tensor.name, label=new_tensor.name)
                log.debug(
                    f'NEW EDGE: {new_node["label"]} -> {next_node["label"]} {self.graph.tensor_map[out_edge["name"]]}'
                )
                op = next_node['op']
                for idx in range(len(op.inputs)):
                    if op.inputs[idx].name == transpose['op'].outputs[0].name:
                        op.inputs[idx] = new_tensor

            remove_ids.append(transpose.index)

        # Delete constant transpose nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def transpose_to_reshape_pass(self):
        filtered_nodes = self.graph.graph.vs.select(
            functools.partial(is_transformable_transpose_node, graph_converter=self.graph.graph)
        )

        # Collect actions for the transformable transpose nodes
        actions = []
        for node in filtered_nodes:
            original_op = node['op']
            output_shape = np.array(original_op.outputs[0].shape, dtype='int32')
            shape_tensor = self.create_attr_tensor(output_shape)
            new_op = tfl.ReshapeOperator(original_op.inputs, original_op.outputs, output_shape)
            node['op'] = new_op
            node['node_type'] = ExtendedOperator.RESHAPE
            node['label'] = new_op.type_name()
            actions.append((self.graph.replace_operator_input, (node, 1, shape_tensor)))

        # Process actions
        for func, args in actions:
            node = args[0]
            func(*args)

    @class_conditional(lambda self: self.level >= GraphOptimizer.FOLD_BUFFER)
    def fold_reshape_buffer(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_constant_reshape_fusable_edge, graph_converter=self.graph.graph)
        )
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for constant, reshape, tensor in filtered_pairs:
            # Calculate the output of the transposed constant nodes
            constant_tensor = reshape['op'].inputs[0].tensor
            shape_tensor = reshape['op'].inputs[1].tensor
            new_constant = np.reshape(constant_tensor, shape_tensor)
            new_tensor = self.create_attr_tensor(new_constant, quantization=reshape['op'].inputs[0].quantization)
            new_node = self.graph.add_nodes([new_tensor])[0]

            # For each node that is next of a constant transpose node, we connect it with the new constant node
            for out_edge in reshape.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                self.graph.graph.add_edge(new_node, next_node, name=new_tensor.name, label=new_tensor.name)
                log.debug(
                    f'NEW EDGE: {new_node["label"]} -> {next_node["label"]} {self.graph.tensor_map[out_edge["name"]]}'
                )
                op = next_node['op']
                for idx in range(len(op.inputs)):
                    if op.inputs[idx].name == reshape['op'].outputs[0].name:
                        op.inputs[idx] = new_tensor

            remove_ids.append(reshape.index)

        # Delete constant transpose nodes
        self.graph.graph.delete_vertices(remove_ids)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def remove_noop_pass(self, branch: bool = False):
        edges = self.graph.graph.es.select(
            functools.partial(is_ending_with_noop_edge, graph_converter=self.graph.graph, branch=branch)
        )
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        if not branch:
            filtered_pairs = fuse_connected_edges(filtered_pairs)

        elinimate_sequences(self.graph, filtered_pairs)

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_wrapped_reshape_within_transpose_pass(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_wrapped_reshape_within_transpose_edge, graph_converter=self.graph.graph)
        )
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        fused_pairs = fuse_connected_edges(filtered_pairs)

        # Only TRANSPOSE->RESHAPE->TRANSPOSE is supported here
        filtered_pairs = []
        for seq in fused_pairs:
            seq_len = len(seq)
            transpose_first = seq[0]['node_type'] == ExtendedOperator.TRANSPOSE
            if seq_len >= 3 and transpose_first:
                filtered_pairs.append(seq[:3])
            elif seq_len >= 4:
                filtered_pairs.append(seq[1:4])

        def _skip_pred(seq):
            mid_node = seq[1]
            orig_shape = mid_node['op'].inputs[0].shape
            new_shape = mid_node['op'].outputs[0].shape

            if not is_simple_reshape(orig_shape, new_shape):
                return True

            new_perm = fuse_transpose_perms_extended(seq)
            return (new_perm != np.sort(new_perm)).any()

        def _remove_last_pred(seq):
            orig_tensor = seq[0]['op'].inputs[0].tensor
            return False, (seq[2], orig_tensor)

        def _remove_last_action(first_node, last_node, custom_data):
            # Set final shape to the first reshape node
            last_trans, orig_tensor = custom_data
            actions = []
            original_op = last_trans['op']
            output_shape = np.array(original_op.outputs[0].shape, dtype='int32')
            shape_tensor = self.create_attr_tensor(output_shape)
            new_op = tfl.ReshapeOperator(original_op.inputs, original_op.outputs, output_shape)
            last_trans['op'] = new_op
            last_trans['node_type'] = ExtendedOperator.RESHAPE
            last_trans['label'] = new_op.type_name()
            new_op.inputs[0].tensor = orig_tensor
            new_op.inputs[0].shape = new_op.inputs[0].tensor.shape
            actions.append((self.graph.replace_operator_input, (last_trans, 1, shape_tensor)))
            return actions

        elinimate_sequences(self.graph, filtered_pairs, True, None, _remove_last_pred, _remove_last_action, _skip_pred)

    @class_conditional(lambda self: self.level >= GraphOptimizer.BRANCH_OPTIMIZE)
    def branch_reshape_expand_pass(self):
        edges = self.graph.graph.es.select(functools.partial(is_reshape_branch_edge, graph_converter=self.graph.graph))
        branch_reshape_nodes = list(set(self.graph.graph.vs[edge.source] for edge in edges))

        def _new_reshape(node: ig.Vertex, prev_node: ig.Vertex, next_node: ig.Vertex):
            actions = []

            op = node['op']
            op_out = op.outputs[0]
            op_shape = op.inputs[1]

            prev_idx = prev_node['outputs'].index(op.inputs[0].name)
            if prev_node['node_type'] == ExtendedOperator.INPUT_NODE:
                prev_out = self.graph.tensor_map[op.inputs[0].name]
            else:
                prev_op = prev_node['op']
                prev_out = prev_op.outputs[prev_idx]

            new_tensor = self.create_transform_tensor(op_out.tensor.copy(), quantization=op_out.quantization)
            new_shape = self.create_attr_tensor(op_shape.tensor.copy())
            new_op = tfl.ReshapeOperator([prev_out, new_shape], [new_tensor], new_shape.tensor)
            new_op.extra_hints.update(op.extra_hints)
            self.graph.add_operator(new_op)

            next_indices = []
            for i, t in enumerate(next_node['op'].inputs):
                if t.name == op_out.name:
                    actions.append((self.graph.replace_operator_input, (next_node, i, new_tensor)))
                    next_indices.append(i)

            assert len(next_indices) > 0, f'{op_out.name} not in {[t.name for t in next_node["op"].inputs]}'

            return actions

        expand_op_outputs_in_branches(branch_reshape_nodes, _new_reshape, self.graph)

    @class_conditional(lambda self: self.level >= GraphOptimizer.BRANCH_OPTIMIZE)
    def branch_transpose_expand_pass(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_transpose_branch_edge, graph_converter=self.graph.graph)
        )
        branch_transpose_nodes = list(set(self.graph.graph.vs[edge.source] for edge in edges))

        def _new_transpose(node: ig.Vertex, prev_node: ig.Vertex, next_node: ig.Vertex):
            actions = []

            op = node['op']
            op_out = op.outputs[0]
            op_perm = op.inputs[1]

            prev_idx = prev_node['outputs'].index(op.inputs[0].name)
            if prev_node['node_type'] in (ExtendedOperator.INPUT_NODE, ExtendedOperator.CONSTANT_NODE):
                prev_out = self.graph.tensor_map[op.inputs[0].name]
            else:
                prev_op = prev_node['op']
                prev_out = prev_op.outputs[prev_idx]

            new_tensor = self.create_transform_tensor(op_out.tensor.copy(), quantization=op_out.quantization)
            new_perm = self.create_attr_tensor(op_perm.tensor.copy())
            new_op = tfl.TransposeOperator([prev_out, new_perm], [new_tensor])
            new_op.extra_hints.update(op.extra_hints)
            self.graph.add_operator(new_op)

            next_indices = []
            for i, t in enumerate(next_node['op'].inputs):
                if t.name == op_out.name:
                    actions.append((self.graph.replace_operator_input, (next_node, i, new_tensor)))
                    next_indices.append(i)

            assert len(next_indices) > 0, f'{op_out.name} not in {[t.name for t in next_node["op"].inputs]}'

            return actions

        expand_op_outputs_in_branches(branch_transpose_nodes, _new_transpose, self.graph)

    @class_conditional(lambda self: self.level >= GraphOptimizer.BRANCH_OPTIMIZE, 0)
    def elementwise_reshape_transpose_passthrough_pass(self) -> int:
        edges = self.graph.graph.es.select(
            functools.partial(is_transpose_reshape_op_edge, graph_converter=self.graph.graph)
        )
        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.TRANSPOSE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        num_actions = 0
        for node in unique_nodes:
            op = node['op']
            input_indices = op_input_indices(op)
            l_shape = op.inputs[0].shape
            r_shape = op.outputs[0].shape
            if len(l_shape) == 0 or len(r_shape) == 0:
                continue
            l_map, r_map, _, _ = reshape_mapping(l_shape, r_shape)
            mode = None
            need_chain = False
            for l_val, r_val in zip(l_map, r_map):
                if len(l_val) > 1 and len(r_val) == 1:
                    if mode in (None, 'up'):
                        mode = 'up'
                    else:
                        mode = '?'
                        break
                elif len(r_val) > 1 and len(l_val) == 1:
                    if mode in (None, 'down'):
                        mode = 'down'
                    else:
                        mode = '?'
                        break
                elif len(r_val) > 1 and len(l_val) > 1:
                    if len(r_val) != len(l_val) or r_val != l_val:
                        # TODO: Support this case
                        mode = '?'
                        break
                    else:
                        need_chain = True

            if mode is None:
                mode = 'down'

            # TODO: Support multi-multi mappings
            if mode == '?':
                continue

            check_consecutive_indices = []
            if need_chain:
                new_l_map = []
                new_r_map = []
                for l_val, r_val in zip(l_map, r_map):
                    if len(l_val) > 1 and len(r_val) > 1:
                        if mode == 'down':
                            check_consecutive_indices.append(l_val)
                        else:
                            check_consecutive_indices.append(r_val)
                        for l_item in l_val:
                            new_l_map.append([l_item])
                        for r_item in r_val:
                            new_r_map.append([r_item])
                    else:
                        new_l_map.append(l_val)
                        new_r_map.append(r_val)

                l_map = new_l_map
                r_map = new_r_map

            prev_nodes = []
            cand_perms = dict()
            cand_rev_perms = dict()
            prev_output_indices = []
            num_constant_nodes = 0
            prev_hints = set()
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

                if prev_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    if mode == 'down':
                        perm = tuple(prev_node['op'].inputs[1].tensor.tolist())
                        cand_perms.setdefault(perm, 0)
                        cand_perms[perm] += 1
                    elif mode == 'up':
                        perm = tuple(np.argsort(prev_node['op'].inputs[1].tensor).tolist())
                        cand_rev_perms.setdefault(perm, 0)
                        cand_rev_perms[perm] += 1
                    if 'direction' in prev_node['op'].extra_hints:
                        prev_hints.add(prev_node['op'].extra_hints['direction'])

                if prev_node['node_type'] == ExtendedOperator.CONSTANT_NODE:
                    num_constant_nodes += 1

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'up' in prev_hints:
                continue

            next_nodes = []
            next_edges = []
            out_nodes = []
            next_hints = set()
            for edge in node.out_edges():
                if edge.index in remove_edges:
                    continue
                next_node = self.graph.graph.vs[edge.target]

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    out_nodes.append(next_node)
                else:
                    next_nodes.append(next_node)
                    next_edges.append(edge)

                if next_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    if mode == 'down':
                        perm = tuple(np.argsort(next_node['op'].inputs[1].tensor).tolist())
                        cand_rev_perms.setdefault(perm, 0)
                        cand_rev_perms[perm] += 1
                    elif mode == 'up':
                        perm = tuple(next_node['op'].inputs[1].tensor.tolist())
                        cand_perms.setdefault(perm, 0)
                        cand_perms[perm] += 1
                    if 'direction' in next_node['op'].extra_hints:
                        next_hints.add(next_node['op'].extra_hints['direction'])

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'down' in next_hints:
                continue

            cur_transpose_size = sum(cand_perms.values()) + sum(cand_rev_perms.values())
            new_transpose_size = len(prev_nodes) + len(next_nodes) - sum(cand_perms.values()) - num_constant_nodes

            # Skip if the number of transpose nodes is not decreasing
            if len(cand_perms) == 0 or len(next_nodes) == 0 or new_transpose_size > cur_transpose_size:
                continue
            elif new_transpose_size == cur_transpose_size:
                skip = True
                if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED:
                    if 'down' in prev_hints or 'up' in next_hints:
                        skip = False

            perm = max(cand_perms.items(), key=lambda x: x[1])[0]
            perm_arr = np.array(perm, dtype='int32')

            skip = False
            for check_idx in check_consecutive_indices:
                if mode == 'down':
                    target_idx = perm_arr[check_idx]
                elif mode == 'up':
                    perm_sorter = perm_arr.argsort()
                    target_idx = perm_sorter[np.searchsorted(perm_arr, check_idx, sorter=perm_sorter)]
                normalized_src = [x - check_idx[0] for x in check_idx]
                normalized_tgt = [x - target_idx[0] for x in target_idx]
                if normalized_src != normalized_tgt:
                    skip = True
                    break

            if skip:
                continue

            num_actions += 1

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for n in out_nodes:
                del self.graph.tensor_map[n['outputs'][0]]
                del self.graph.tensor_node_map[n['outputs'][0]]

            if mode == 'down':
                inv_perm_arr = np.argsort(perm_arr).astype('int32')
                l_dict = dict(zip([x[0] for x in l_map], r_map))
                indices = map(lambda x: l_dict[x], inv_perm_arr.tolist())
                inv_post_perm = list(itertools.chain.from_iterable(indices))
                inv_post_perm_arr = np.array(inv_post_perm, dtype='int32')
                post_perm_arr = np.argsort(inv_post_perm_arr).astype('int32')
            elif mode == 'up':
                r_dict = dict(zip([x[0] for x in r_map], l_map))
                indices = map(lambda x: r_dict[x], perm)
                inv_perm = list(itertools.chain.from_iterable(indices))
                inv_perm_arr = np.array(inv_perm, dtype='int32')
                post_perm_arr = np.argsort(perm_arr).astype('int32')
                inv_post_perm_arr = np.argsort(post_perm_arr).astype('int32')

            for prev_node, prev_idx, next_idx in zip(prev_nodes, input_indices, prev_output_indices):
                if prev_node['op'] is None:
                    prev_out = self.graph.tensor_map[prev_node['outputs'][0]]
                else:
                    prev_out = prev_node['op'].outputs[next_idx]
                perm_tensor = self.create_attr_tensor(inv_perm_arr)
                prev_new_out = self.create_transform_tensor(
                    np.transpose(prev_out.tensor, inv_perm_arr), quantization=prev_out.quantization
                )
                transpose_op = tfl.TransposeOperator([prev_out, perm_tensor], [prev_new_out])
                transpose_op.extra_hints['direction'] = 'up'
                self.graph.add_operator(transpose_op)
                actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                perm_tensor = self.create_attr_tensor(post_perm_arr)
                new_out = self.create_transform_tensor(
                    np.transpose(op_out.tensor, inv_post_perm_arr), quantization=op_out.quantization
                )

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                transpose_op = tfl.TransposeOperator([new_out, perm_tensor], [op_out])
                transpose_op.extra_hints['direction'] = 'down'
                self.graph.add_operator(transpose_op)

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

            # OP specific dim handling logic
            old_shape = op.inputs[1].tensor
            new_shape = self.create_attr_tensor(old_shape[inv_post_perm_arr])
            actions.append((self.graph.replace_operator_input, (node, 1, new_shape, True)))
            op.newShape = new_shape.tensor

            for edge in next_edges:
                source = tensor_node_dict[edge['name']]
                self.graph.graph.add_edge(source, edge.target_vertex, name=edge['name'], label=edge['name'])

        # Process actions
        ids = []
        for func, args in actions:
            node = args[0]
            res = func(*args)
            if res is not None:
                ids.extend(res)

        remove_edges = list(set(remove_edges + ids))

        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

        return num_actions

    @class_conditional(lambda self: self.rewrite_quantizable)
    def elementwise_op_quantize_passthrough_pass(self):
        edges = self.graph.graph.es.select(
            functools.partial(is_quantize_elementwise_op_edge, graph_converter=self.graph.graph)
        )
        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.DEQUANTIZE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        for node in unique_nodes:
            op = node['op']
            input_indices = op_input_indices(op)

            prev_nodes = []
            q_tensors = dict()
            prev_output_indices = []
            skip_names = []
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

                if prev_node['node_type'] == ExtendedOperator.DEQUANTIZE:
                    q_tensors[prev_node_name] = prev_node['op'].inputs[0]

                if prev_node['node_type'] == ExtendedOperator.CONSTANT_NODE and prev_node_name in self.graph.q_mapping:
                    skip_names.append(prev_node_name)

            next_nodes = []
            next_edges = []
            out_nodes = []
            for edge in node.out_edges():
                if edge.index in remove_edges:
                    continue
                next_node = self.graph.graph.vs[edge.target]

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    out_nodes.append(next_node)
                else:
                    next_nodes.append(next_node)
                    next_edges.append(edge)

                if next_node['node_type'] == ExtendedOperator.QUANTIZE:
                    skip = False
                    name = next_node['op'].inputs[0].name
                    q_tensor = next_node['op'].outputs[0]
                    assert q_tensor.quantization is not None
                    if node['node_type'] in (
                        ExtendedOperator.BATCH_MATMUL,
                        ExtendedOperator.ABS,
                        ExtendedOperator.RSQRT,
                    ):
                        if q_tensor.dtype not in (np.dtype('int8'), np.dtype('int16')):
                            skip = True
                    elif node['node_type'] == ExtendedOperator.DIV:
                        if q_tensor.dtype != np.dtype('uint8'):
                            skip = True
                    elif node['node_type'] == ExtendedOperator.SOFTMAX:
                        if q_tensor.dtype == np.dtype('int8'):
                            if (
                                abs(q_tensor.quantization.scale - 1.0 / 256) > 0.001 * 1.0 / 256
                                or q_tensor.quantization.zero_point != -128
                            ):
                                skip = True
                        elif q_tensor.dtype == np.dtype('int16'):
                            if (
                                abs(q_tensor.quantization.scale - 1.0 / 32768) > 0.001 * 1.0 / 32768
                                or q_tensor.quantization.zero_point != 0
                            ):
                                skip = True
                        elif q_tensor.dtype == np.dtype('uint8'):
                            if (
                                abs(q_tensor.quantization.scale - 1.0 / 256) > 0.001 * 1.0 / 256
                                or q_tensor.quantization.zero_point != 0
                            ):
                                log.warning(
                                    'On some chips, only softmax with scale=1.0/256 and zero_point=0 is supported'
                                )
                        else:
                            skip = True
                    elif node['node_type'] == ExtendedOperator.LOG_SOFTMAX:
                        if q_tensor.dtype == np.dtype('int8'):
                            if q_tensor.quantization.scale != 16.0 / 256 or q_tensor.quantization.zero_point != 127:
                                skip = True
                        elif q_tensor.dtype == np.dtype('uint8'):
                            if q_tensor.quantization.scale != 16.0 / 256 or q_tensor.quantization.zero_point != 255:
                                skip = True
                        else:
                            skip = True

                    if not skip:
                        q_tensors[name] = q_tensor

            cur_transpose_size = len(q_tensors)
            new_transpose_size = len(prev_nodes) + len(next_nodes) - len(skip_names)

            # Skip if the number of [de]quantize nodes is not decreasing
            if len(next_nodes) == 0 or new_transpose_size > cur_transpose_size:
                continue

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for n in out_nodes:
                del self.graph.tensor_map[n['outputs'][0]]
                del self.graph.tensor_node_map[n['outputs'][0]]

            tensor_node_dict = {}
            for prev_node, prev_idx, next_idx in zip(prev_nodes, input_indices, prev_output_indices):
                if prev_node['op'] is None:
                    prev_out = self.graph.tensor_map[prev_node['outputs'][0]]
                else:
                    prev_out = prev_node['op'].outputs[next_idx]
                if prev_out.name in tensor_node_dict:
                    prev_new_out, skip = tensor_node_dict[prev_out.name]
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True, skip)))
                    skip += 1
                    tensor_node_dict[prev_out.name] = (prev_new_out, skip)
                else:
                    if prev_out.name in skip_names:
                        prev_new_out = self.graph.q_mapping[prev_out.name]
                        self.graph.add_nodes([prev_new_out])
                        tensor_node_dict[prev_out.name] = (prev_new_out, 1)
                        actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))
                    else:
                        prev_new_out = self.create_transform_tensor(
                            q_tensors[prev_out.name].tensor, quantization=q_tensors[prev_out.name].quantization
                        )
                        tensor_node_dict[prev_out.name] = (prev_new_out, 1)
                        self.graph.add_operator(tfl.QuantizeOperator([prev_out], [prev_new_out]))
                        actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                new_out = self.create_transform_tensor(
                    q_tensors[op_out.name].tensor, quantization=q_tensors[op_out.name].quantization
                )

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                self.graph.add_operator(tfl.DequantizeOperator([new_out], [op_out]))

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

            for edge in next_edges:
                source = tensor_node_dict[edge['name']]
                self.graph.graph.add_edge(source, edge.target_vertex, name=edge['name'], label=edge['name'])

        # Process actions
        ids = []
        for func, args in actions:
            node = args[0]
            res = func(*args)
            if res is not None:
                ids.extend(res)

        remove_edges = list(set(remove_edges + ids))

        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

    @class_conditional(lambda self: self.level >= GraphOptimizer.BRANCH_OPTIMIZE, 0)
    def elementwise_op_transpose_passthrough_pass(self, quantizable_ops_only: bool = False) -> int:
        edges = self.graph.graph.es.select(
            functools.partial(
                is_transpose_elementwise_op_edge,
                graph_converter=self.graph.graph,
                quantizable_ops_only=quantizable_ops_only,
            )
        )

        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        if quantizable_ops_only:
            all_edges = self.graph.graph.es.select(
                functools.partial(
                    is_transpose_elementwise_op_edge,
                    graph_converter=self.graph.graph,
                    quantizable_ops_only=False,
                )
            )

            all_pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in all_edges)

            forward_d = dict(all_pairs)
            backward_d = {v: k for k, v in forward_d.items()}

            filtered_nodes = []
            for s, e in pairs:
                if s['node_type'] == ExtendedOperator.TRANSPOSE:
                    pn = backward_d.get(s, None)
                    if pn is not None:
                        filtered_nodes.append(pn)
                    else:
                        log.warning('Cannot passthrough transpose upward around requantizable ops')
                else:
                    pn = forward_d.get(e, None)
                    if pn is not None:
                        filtered_nodes.append(pn)
                    else:
                        log.warning('Cannot passthrough transpose downward around requantizable ops')
        else:
            filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.TRANSPOSE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        num_actions = 0
        for node in unique_nodes:
            op = node['op']
            input_indices = op_input_indices(op)

            prev_nodes = []
            cand_perms = dict()
            prev_output_indices = []
            num_constant_nodes = 0
            num_reshape_transpose = 0
            prev_hints = set()
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

                if prev_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    perm = tuple(prev_node['op'].inputs[1].tensor.tolist())

                    if node['node_type'] == ExtendedOperator.PACK:
                        perm = [i if i < op.axis else i + 1 for i in perm]
                        perm.insert(op.axis, op.axis)
                        perm = tuple(perm)

                    cand_perms.setdefault(perm, 0)
                    cand_perms[perm] += 1
                    if 'direction' in prev_node['op'].extra_hints:
                        prev_hints.add(prev_node['op'].extra_hints['direction'])

                if prev_node['node_type'] == ExtendedOperator.CONSTANT_NODE:
                    num_constant_nodes += 1

                if prev_node['node_type'] == ExtendedOperator.RESHAPE:
                    prev_prev_node_name = self.graph.tensor_node_map[prev_node['op'].inputs[0].name]
                    prev_prev_node = self.graph.graph.vs.find(name=prev_prev_node_name)
                    if prev_prev_node['node_type'] == ExtendedOperator.TRANSPOSE:
                        num_reshape_transpose += 1
                        if 'direction' in prev_prev_node['op'].extra_hints:
                            prev_hints.add(prev_prev_node['op'].extra_hints['direction'])

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'up' in prev_hints:
                continue

            next_nodes = []
            next_edges = []
            out_nodes = []
            skip_names = []
            next_hints = set()
            for edge in node.out_edges():
                if edge.index in remove_edges:
                    continue
                next_node = self.graph.graph.vs[edge.target]

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    out_nodes.append(next_node)
                elif next_node['node_type'] == ExtendedOperator.UNUSED_NODE:
                    skip_names.append(edge['label'])
                else:
                    next_nodes.append(next_node)
                    next_edges.append(edge)

                if next_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    perm = tuple(np.argsort(next_node['op'].inputs[1].tensor).tolist())

                    if node['node_type'] == ExtendedOperator.UNPACK:
                        perm = [i if i < op.axis else i + 1 for i in perm]
                        perm.insert(op.axis, op.axis)
                        perm = tuple(perm)

                    cand_perms.setdefault(perm, 0)
                    cand_perms[perm] += 1
                    if 'direction' in next_node['op'].extra_hints:
                        next_hints.add(next_node['op'].extra_hints['direction'])

                if next_node['node_type'] == ExtendedOperator.RESHAPE:
                    o_nodes = [e.target_vertex for e in next_node.out_edges()]
                    if len(o_nodes) == 1 and o_nodes[0]['node_type'] == ExtendedOperator.TRANSPOSE:
                        num_reshape_transpose += 1
                        if 'direction' in o_nodes[0]['op'].extra_hints:
                            next_hints.add(o_nodes[0]['op'].extra_hints['direction'])

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'down' in next_hints:
                continue

            cur_transpose_size = sum(cand_perms.values()) + num_reshape_transpose
            new_transpose_size = (
                len(prev_nodes) + len(next_nodes) - num_constant_nodes - cur_transpose_size + num_reshape_transpose
            )

            # Skip if the number of transpose nodes is not decreasing
            if len(next_nodes) == 0 or new_transpose_size > cur_transpose_size:
                continue
            elif new_transpose_size == cur_transpose_size:
                skip = True
                if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED:
                    if 'down' in prev_hints or 'up' in next_hints:
                        skip = False
                if skip:
                    continue

            num_actions += 1

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for n in out_nodes:
                del self.graph.tensor_map[n['outputs'][0]]
                del self.graph.tensor_node_map[n['outputs'][0]]

            perm = max(cand_perms.items(), key=lambda x: x[1])[0]
            perm_arr = np.array(perm, dtype='int32')
            inv_perm_arr = np.argsort(perm_arr).astype('int32')

            if node['node_type'] == ExtendedOperator.UNPACK:
                inv_perm_arr_post = inv_perm_arr[inv_perm_arr != op.axis]
                inv_perm_arr_post[inv_perm_arr_post > op.axis] -= 1

                perm_arr_post = np.argsort(inv_perm_arr_post).astype('int32')
            elif node['node_type'] == ExtendedOperator.PACK:
                perm_arr_post = perm_arr
                inv_perm_arr_post = inv_perm_arr

                perm_arr = perm_arr_post[perm_arr_post != op.axis]
                perm_arr[perm_arr > op.axis] -= 1

                inv_perm_arr = np.argsort(perm_arr).astype('int32')
            else:
                perm_arr_post = perm_arr
                inv_perm_arr_post = inv_perm_arr

            tensor_node_dict = {}
            for prev_node, prev_idx, next_idx in zip(prev_nodes, input_indices, prev_output_indices):
                if prev_node['op'] is None:
                    prev_out = self.graph.tensor_map[prev_node['outputs'][0]]
                else:
                    prev_out = prev_node['op'].outputs[next_idx]
                if prev_out.name in tensor_node_dict:
                    prev_new_out, skip = tensor_node_dict[prev_out.name]
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True, skip)))
                    skip += 1
                    tensor_node_dict[prev_out.name] = (prev_new_out, skip)
                else:
                    perm_tensor = self.create_attr_tensor(inv_perm_arr)
                    if len(prev_out.shape) != perm_tensor.tensor.size:
                        new_shape = [1] * (perm_tensor.tensor.size - len(prev_out.shape)) + list(prev_out.shape)
                        prev_out_reshaped = self.create_transform_tensor(
                            np.reshape(prev_out.tensor, new_shape), quantization=prev_out.quantization
                        )
                        new_shape_tensor = self.create_attr_tensor(np.array(new_shape, dtype='int32'))
                        self.graph.add_operator(
                            tfl.ReshapeOperator([prev_out, new_shape_tensor], [prev_out_reshaped], new_shape)
                        )
                        prev_out = prev_out_reshaped
                    prev_new_out = self.create_transform_tensor(
                        np.transpose(prev_out.tensor, inv_perm_arr), quantization=prev_out.quantization
                    )
                    tensor_node_dict[prev_out.name] = (prev_new_out, 1)
                    transpose_op = tfl.TransposeOperator([prev_out, perm_tensor], [prev_new_out])
                    transpose_op.extra_hints['direction'] = 'up'
                    self.graph.add_operator(transpose_op)
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                # For unused tensors, we perform inplace shape updates
                if op_out.name in skip_names:
                    orig_shape = np.array(op_out.shape, dtype='int32')
                    new_shape = orig_shape[inv_perm_arr]
                    op_out.shape = tuple(new_shape.tolist())
                    continue

                perm_tensor = self.create_attr_tensor(perm_arr_post)
                new_out = self.create_transform_tensor(
                    np.transpose(op_out.tensor, inv_perm_arr_post), quantization=op_out.quantization
                )

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                transpose_op = tfl.TransposeOperator([new_out, perm_tensor], [op_out])
                transpose_op.extra_hints['direction'] = 'down'
                self.graph.add_operator(transpose_op)

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

            # OP specific dim handling logic
            if node['node_type'] in (ExtendedOperator.CONCATENATION, ExtendedOperator.GATHER):
                old_axis = op.axis
                new_axis = np.where(inv_perm_arr == old_axis)[0][0]
                op.axis = new_axis
            elif node['node_type'] == ExtendedOperator.SPLIT_V:
                old_dim = op.inputs[2].tensor
                new_dim = np.where(inv_perm_arr == old_dim)[0][0]
                new_dim_tensor = self.create_attr_tensor(np.array([new_dim], dtype='int32'))
                actions.append((self.graph.replace_operator_input, (node, 2, new_dim_tensor, True)))
            elif node['node_type'] == ExtendedOperator.SPLIT:
                old_dim = op.inputs[0].tensor
                new_dim = np.where(inv_perm_arr == old_dim)[0][0]
                new_dim_tensor = self.create_attr_tensor(np.array([new_dim], dtype='int32'))
                actions.append((self.graph.replace_operator_input, (node, 0, new_dim_tensor, True)))
            elif node['node_type'] in (
                ExtendedOperator.PAD,
                ExtendedOperator.PADV2,
                ExtendedOperator.MIRROR_PAD,
                ExtendedOperator.TILE,
            ):
                old_pad = op.inputs[1].tensor
                new_pad = self.create_attr_tensor(old_pad[inv_perm_arr])
                actions.append((self.graph.replace_operator_input, (node, 1, new_pad, True)))
            elif node['node_type'] == ExtendedOperator.PRELU:
                old_weight = op.inputs[1].tensor
                if old_weight.ndim != 1:
                    assert old_weight.ndim + 1 == len(inv_perm_arr)
                    new_perm = np.argsort(np.argsort(inv_perm_arr[1:]))
                    new_perm_t = self.create_attr_tensor(np.array(new_perm, dtype='int32'))
                    new_weight = self.create_transform_tensor(np.transpose(old_weight, new_perm))
                    self.graph.add_operator(tfl.TransposeOperator([op.inputs[1], new_perm_t], [new_weight]))
                    actions.append((self.graph.replace_operator_input, (node, 1, new_weight, True)))
            elif node['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE):
                for i, t in enumerate(op.inputs[1:]):
                    new_t = self.create_attr_tensor(t.tensor[inv_perm_arr])
                    actions.append((self.graph.replace_operator_input, (node, i + 1, new_t, True)))
            elif node['node_type'] in (
                ExtendedOperator.SUM,
                ExtendedOperator.ARG_MIN,
                ExtendedOperator.ARG_MAX,
                ExtendedOperator.REDUCE_MIN,
                ExtendedOperator.REDUCE_MAX,
                ExtendedOperator.REDUCE_PROD,
                ExtendedOperator.MEAN,
            ):
                old_axis = op.inputs[1].tensor.tolist()
                new_axis = []
                for t in old_axis:
                    new_t = np.where(inv_perm_arr == t)[0][0]
                    new_axis.append(new_t)
                axis_arr = np.array(new_axis, dtype='int32')
                axis_tensor = self.create_attr_tensor(axis_arr)
                actions.append((self.graph.replace_operator_input, (node, 1, axis_tensor, True)))

            for edge in next_edges:
                source = tensor_node_dict[edge['name']]
                self.graph.graph.add_edge(source, edge.target_vertex, name=edge['name'], label=edge['name'])

        # Process actions
        ids = []
        for func, args in actions:
            node = args[0]
            res = func(*args)
            if res is not None:
                ids.extend(res)

        remove_edges = list(set(remove_edges + ids))

        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

        return num_actions

    @class_conditional(lambda self: self.level >= GraphOptimizer.BRANCH_OPTIMIZE, 0)
    def elementwise_op_reshape_passthrough_pass(self) -> int:
        edges = self.graph.graph.es.select(
            functools.partial(is_reshape_elementwise_op_edge, graph_converter=self.graph.graph)
        )
        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.RESHAPE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        num_actions = 0
        for node in unique_nodes:
            op = node['op']
            dim_indice = op_input_dims(op)
            input_indices = op_input_indices(op)

            prev_nodes = []
            cand_shapes = dict()
            cand_next_shapes = dict()
            prev_output_indices = []
            num_constant_nodes = 0
            prev_hints = set()
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

                if prev_node['node_type'] == ExtendedOperator.CONSTANT_NODE:
                    num_constant_nodes += 1

                if prev_node['node_type'] == ExtendedOperator.RESHAPE:
                    mapping = dict()
                    if not is_simple_reshape(
                        prev_node['op'].inputs[0].shape, prev_node['op'].outputs[0].shape, mapping
                    ):
                        continue

                    new_dim = None
                    if dim_indice is not None:
                        rev_mapping = {v: k for k, v in mapping.items()}
                        if node['node_type'] == ExtendedOperator.PACK:
                            if dim_indice in rev_mapping:
                                tmp_new_dim = rev_mapping[dim_indice]
                            else:
                                if dim_indice - 1 in rev_mapping:
                                    tmp_new_dim = rev_mapping[dim_indice - 1] + 1
                                elif dim_indice + 1 in rev_mapping:
                                    tmp_new_dim = rev_mapping[dim_indice + 1] - 1
                                else:
                                    # TODO: Figure out the rev index
                                    tmp_new_dim = -1
                            tmp_dim_indice = dim_indice
                            new_dim = -1
                            dim_indice = -1
                        else:
                            if dim_indice not in rev_mapping:
                                continue
                            new_dim = rev_mapping[dim_indice]

                    shape = tuple(prev_node['op'].inputs[0].shape)
                    shape = tuple(x if i != new_dim else -1 for i, x in enumerate(shape))
                    if node['node_type'] == ExtendedOperator.PACK and tmp_new_dim >= 0:
                        shape = list(shape)
                        shape.insert(tmp_new_dim, -1)
                        shape = tuple(shape)
                    cand_shapes.setdefault(shape, 0)
                    cand_shapes[shape] += 1

                    next_shape = tuple(prev_node['op'].outputs[0].shape)
                    next_shape = tuple(x if i != dim_indice else -1 for i, x in enumerate(next_shape))
                    if node['node_type'] == ExtendedOperator.PACK:
                        next_shape = list(next_shape)
                        next_shape.insert(tmp_dim_indice, -1)
                        next_shape = tuple(next_shape)
                    cand_next_shapes.setdefault(next_shape, 0)
                    cand_next_shapes[next_shape] += 1

                    if node['node_type'] == ExtendedOperator.PACK:
                        dim_indice = tmp_dim_indice

                    if 'direction' in prev_node['op'].extra_hints:
                        prev_hints.add(prev_node['op'].extra_hints['direction'])

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'up' in prev_hints:
                continue

            next_nodes = []
            next_edges = []
            out_nodes = []
            skip_names = []
            next_hints = set()
            for edge in node.out_edges():
                if edge.index in remove_edges:
                    continue
                next_node = self.graph.graph.vs[edge.target]

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    out_nodes.append(next_node)
                elif next_node['node_type'] == ExtendedOperator.UNUSED_NODE:
                    skip_names.append(edge['label'])
                else:
                    next_nodes.append(next_node)
                    next_edges.append(edge)

                if next_node['node_type'] == ExtendedOperator.RESHAPE:
                    mapping = dict()
                    if not is_simple_reshape(
                        next_node['op'].inputs[0].shape, next_node['op'].outputs[0].shape, mapping
                    ):
                        continue

                    new_dim = None
                    if dim_indice is not None:
                        if node['node_type'] == ExtendedOperator.UNPACK:
                            if dim_indice in mapping:
                                tmp_new_dim = mapping[dim_indice]
                            else:
                                if dim_indice - 1 in mapping:
                                    tmp_new_dim = mapping[dim_indice - 1] + 1
                                elif dim_indice + 1 in mapping:
                                    tmp_new_dim = mapping[dim_indice + 1] - 1
                                else:
                                    # TODO: Figure out the rev index
                                    tmp_new_dim = -1
                            tmp_dim_indice = dim_indice
                            new_dim = -1
                            dim_indice = -1
                        else:
                            if dim_indice not in mapping:
                                continue
                            new_dim = mapping[dim_indice]

                    shape = tuple(next_node['op'].outputs[0].shape)
                    shape = tuple(x if i != new_dim else -1 for i, x in enumerate(shape))
                    if node['node_type'] == ExtendedOperator.UNPACK and tmp_new_dim >= 0:
                        shape = list(shape)
                        shape.insert(tmp_new_dim, -1)
                        shape = tuple(shape)
                    cand_shapes.setdefault(shape, 0)
                    cand_shapes[shape] += 1

                    next_shape = tuple(next_node['op'].inputs[0].shape)
                    next_shape = tuple(x if i != dim_indice else -1 for i, x in enumerate(next_shape))
                    if node['node_type'] == ExtendedOperator.UNPACK:
                        next_shape = list(next_shape)
                        next_shape.insert(tmp_dim_indice, -1)
                        next_shape = tuple(next_shape)
                    cand_next_shapes.setdefault(next_shape, 0)
                    cand_next_shapes[next_shape] += 1

                    if node['node_type'] == ExtendedOperator.UNPACK:
                        dim_indice = tmp_dim_indice

                    if 'direction' in next_node['op'].extra_hints:
                        next_hints.add(next_node['op'].extra_hints['direction'])

            if len(cand_shapes) == 0:
                continue

            if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED and 'down' in next_hints:
                continue

            cur_reshape_size = max(cand_shapes.values())
            cur_next_reshape_size = max(cand_next_shapes.values())
            full_size = len(prev_nodes) + len(next_nodes)

            if cur_reshape_size != cur_next_reshape_size:
                continue

            new_reshape_size = full_size - cur_reshape_size - num_constant_nodes

            # Skip if not wrapped by reshapes
            if (
                len(next_nodes) == 0 or new_reshape_size > cur_reshape_size
            ):  # cur_reshape_size < full_size or cur_next_reshape_size < full_size:
                continue
            elif new_reshape_size == cur_reshape_size:
                skip = True
                if self.level >= GraphOptimizer.BRANCH_OPTIMIZE_EXTENDED:
                    if 'down' in prev_hints or 'up' in next_hints:
                        skip = False
                if skip:
                    continue

            num_actions += 1

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for n in out_nodes:
                del self.graph.tensor_map[n['outputs'][0]]
                del self.graph.tensor_node_map[n['outputs'][0]]

            prev_shape = max(cand_shapes.items(), key=lambda x: x[1])[0]
            next_shape = max(cand_next_shapes.items(), key=lambda x: x[1])[0]

            tensor_node_dict = {}
            for prev_node, prev_idx, next_idx in zip(prev_nodes, input_indices, prev_output_indices):
                if prev_node['op'] is None:
                    prev_out = self.graph.tensor_map[prev_node['outputs'][0]]
                else:
                    prev_out = prev_node['op'].outputs[next_idx]
                if prev_out.name in tensor_node_dict:
                    prev_new_out, skip = tensor_node_dict[prev_out.name]
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True, skip)))
                    skip += 1
                    tensor_node_dict[prev_out.name] = (prev_new_out, skip)
                else:
                    if node['node_type'] == ExtendedOperator.PACK:
                        tmp_prev_shape = prev_shape
                        prev_shape = [i for i in prev_shape if i != -1]
                    prev_shape_aligned = prev_shape
                    if np.prod(prev_out.shape) != np.prod(prev_shape):
                        new_prev_shape = prev_out.shape
                        if len(prev_out.shape) < len(next_shape):
                            new_prev_shape = [1] * (len(next_shape) - len(prev_out.shape)) + list(prev_out.shape)
                        mapping = {}
                        is_simple_reshape(prev_shape, next_shape, mapping)
                        prev_shape_aligned = np.ones(len(prev_shape), dtype='int32')
                        for pi, ni in mapping.items():
                            prev_shape_aligned[pi] = new_prev_shape[ni]

                    prev_new_out = self.create_transform_tensor(
                        np.reshape(prev_out.tensor, prev_shape_aligned), quantization=prev_out.quantization
                    )
                    tensor_node_dict[prev_out.name] = (prev_new_out, 1)
                    shape_tensor = self.create_attr_tensor(np.array(prev_new_out.shape, dtype='int32'))
                    reshape_op = tfl.ReshapeOperator(
                        [prev_out, shape_tensor], [prev_new_out], newShape=shape_tensor.tensor
                    )
                    reshape_op.extra_hints['direction'] = 'up'
                    self.graph.add_operator(reshape_op)
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

                    if node['node_type'] == ExtendedOperator.PACK:
                        prev_shape = tmp_prev_shape

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                if node['node_type'] == ExtendedOperator.UNPACK:
                    tmp_prev_shape = prev_shape
                    prev_shape = [i for i in prev_shape if i != -1]

                # For unused tensors, we perform inplace shape updates
                if op_out.name in skip_names:
                    new_shape = np.reshape(op_out.tensor, prev_shape).shape
                    op_out.shape = tuple(new_shape)

                    if node['node_type'] == ExtendedOperator.UNPACK:
                        prev_shape = tmp_prev_shape

                    continue

                new_out = self.create_transform_tensor(
                    np.reshape(op_out.tensor, prev_shape), quantization=op_out.quantization
                )
                shape_tensor = self.create_attr_tensor(np.array(op_out.shape, dtype='int32'))

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                reshape_op = tfl.ReshapeOperator([new_out, shape_tensor], [op_out], shape_tensor.tensor)
                reshape_op.extra_hints['direction'] = 'down'
                self.graph.add_operator(reshape_op)

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

                if node['node_type'] == ExtendedOperator.UNPACK:
                    prev_shape = tmp_prev_shape

            # OP specific dim handling logic
            if node['node_type'] in (
                ExtendedOperator.CONCATENATION,
                ExtendedOperator.GATHER,
                ExtendedOperator.UNPACK,
                ExtendedOperator.PACK,
            ):
                new_axis = prev_shape.index(-1)
                op.axis = new_axis
            elif node['node_type'] == ExtendedOperator.SPLIT_V:
                new_dim = prev_shape.index(-1)
                new_dim_tensor = self.create_attr_tensor(np.array([new_dim], dtype='int32'))
                actions.append((self.graph.replace_operator_input, (node, 2, new_dim_tensor, True)))
            elif node['node_type'] == ExtendedOperator.SPLIT:
                new_dim = prev_shape.index(-1)
                new_dim_tensor = self.create_attr_tensor(np.array([new_dim], dtype='int32'))
                actions.append((self.graph.replace_operator_input, (node, 0, new_dim_tensor, True)))
            elif node['node_type'] in (ExtendedOperator.PAD, ExtendedOperator.PADV2, ExtendedOperator.MIRROR_PAD):
                old_pad = op.inputs[1].tensor
                new_dim = prev_shape.index(-1)
                old_dim = next_shape.index(-1)
                new_pad = np.zeros((len(prev_shape), 2), dtype='int32')
                new_pad[new_dim, :] = old_pad[old_dim, :]
                new_pad_tensor = self.create_attr_tensor(new_pad)
                actions.append((self.graph.replace_operator_input, (node, 1, new_pad_tensor, True)))
            elif node['node_type'] == ExtendedOperator.PRELU:
                old_weight = op.inputs[1].tensor
                if old_weight.ndim != 1:
                    new_dim = prev_shape.index(-1)
                    old_dim = next_shape.index(-1)
                    new_shape = np.ones(len(prev_shape) - 1, dtype='int32')
                    new_shape[new_dim - 1] = old_weight.shape[old_dim - 1]
                    new_shape_t = self.create_attr_tensor(new_shape)
                    new_weight = self.create_transform_tensor(np.reshape(old_weight, new_shape))
                    self.graph.add_operator(tfl.ReshapeOperator([op.inputs[1], new_shape_t], [new_weight], new_shape))
                    actions.append((self.graph.replace_operator_input, (node, 1, new_weight, True)))
            elif node['node_type'] == ExtendedOperator.SLICE:
                new_dim = prev_shape.index(-1)
                old_dim = next_shape.index(-1)

                new_start = np.zeros(len(prev_shape), dtype='int32')
                new_start[new_dim] = op.inputs[1].tensor[old_dim]
                new_start_t = self.create_attr_tensor(new_start)

                new_size = np.array(prev_shape, dtype='int32')
                new_size[new_dim] = op.inputs[2].tensor[old_dim]
                new_size_t = self.create_attr_tensor(new_size)

                actions.append((self.graph.replace_operator_input, (node, 1, new_start_t, True)))
                actions.append((self.graph.replace_operator_input, (node, 2, new_size_t, True)))
            elif node['node_type'] == ExtendedOperator.STRIDED_SLICE:
                new_dim = prev_shape.index(-1)
                old_dim = next_shape.index(-1)

                new_start = np.zeros(len(prev_shape), dtype='int32')
                new_start[new_dim] = op.inputs[1].tensor[old_dim]
                new_start_t = self.create_attr_tensor(new_start)

                new_end = np.array(prev_shape, dtype='int32')
                new_end[new_dim] = op.inputs[2].tensor[old_dim]
                new_end_t = self.create_attr_tensor(new_end)

                new_stride = np.ones(len(prev_shape), dtype='int32')
                new_stride[new_dim] = op.inputs[3].tensor[old_dim]
                new_stride_t = self.create_attr_tensor(new_stride)

                actions.append((self.graph.replace_operator_input, (node, 1, new_start_t, True)))
                actions.append((self.graph.replace_operator_input, (node, 2, new_end_t, True)))
                actions.append((self.graph.replace_operator_input, (node, 3, new_stride_t, True)))
            elif node['node_type'] == ExtendedOperator.TILE:
                old_shape = op.inputs[1].tensor
                new_dim = prev_shape.index(-1)
                old_dim = next_shape.index(-1)
                new_shape = np.ones(len(prev_shape), dtype='int32')
                new_shape[new_dim] = old_shape[old_dim]
                new_shape_tensor = self.create_attr_tensor(new_shape)
                actions.append((self.graph.replace_operator_input, (node, 1, new_shape_tensor, True)))
            elif node['node_type'] in (
                ExtendedOperator.SUM,
                ExtendedOperator.ARG_MIN,
                ExtendedOperator.ARG_MAX,
                ExtendedOperator.REDUCE_MIN,
                ExtendedOperator.REDUCE_MAX,
                ExtendedOperator.REDUCE_PROD,
                ExtendedOperator.MEAN,
            ):
                new_axis = prev_shape.index(-1)
                axis_arr = np.array([new_axis], dtype='int32')
                axis_tensor = self.create_attr_tensor(axis_arr)
                actions.append((self.graph.replace_operator_input, (node, 1, axis_tensor, True)))
            elif dim_indice is not None:
                raise NotImplementedError(f'{node["node_type"]} has the property `dims` but is not handled')

            for edge in next_edges:
                source = tensor_node_dict[edge['name']]
                self.graph.graph.add_edge(source, edge.target_vertex, name=edge['name'], label=edge['name'])

        # Process actions
        ids = []
        for func, args in actions:
            node = args[0]
            res = func(*args)
            if res is not None:
                ids.extend(res)

        remove_edges = list(set(remove_edges + ids))

        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

        return num_actions

    @class_conditional(lambda self: self.level >= GraphOptimizer.COMMON_OPTIMIZE)
    def fuse_bmm_add_pass(self):
        edges = self.graph.graph.es.select(functools.partial(is_bmm_add_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]
        filtered_pairs = [
            p
            for p in filtered_pairs
            if p[0]['node_type'] != ExtendedOperator.FULLY_CONNECTED
            or len(p[0]['op'].inputs) == 2
            or not np.any(p[0]['op'].inputs[2].tensor)
        ]

        remove_ids = []
        ops = []
        restore_mapping = []
        for bmm, add in filtered_pairs:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in add.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in add['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            ops.append((bmm, add))
            remove_ids.append(bmm.index)
            remove_ids.append(add.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [
            (nodes[0]['op'], nodes[1]['op'])
            for nodes in sorted(ops, key=lambda x: int(re.search(r'\d+', x[1]['name'])[0]))
        ]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for (bmm, add), mapping in zip(sorted_ops, restore_mapping):
            input_tensor = bmm.inputs[0]
            weight_tensor = bmm.inputs[1]
            bias_tensor = add.inputs[1]
            output_tensor = add.outputs[0]

            ops = []

            if isinstance(bmm, tfl.BatchMatmulOperator):
                weight_t = self.create_transform_tensor(np.transpose(weight_tensor.tensor))
                weight_perm = self.create_attr_tensor(np.array([1, 0], dtype='int32'))
                ops.append(tfl.TransposeOperator([weight_tensor, weight_perm], [weight_t]))
            else:
                weight_t = weight_tensor

            keep_dims = output_tensor.tensor.ndim > 2

            ops.append(
                tfl.FullyConnectedOperator(
                    [input_tensor, weight_t, bias_tensor],
                    [output_tensor],
                    fusedActivationFunction=add.fusedActivationFunction,
                    keepNumDims=keep_dims,
                )
            )

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    @class_conditional(lambda self: self.max_transpose_dims > 0)
    def lower_transpose_dim_pass(self):
        vertices = self.graph.graph.vs.select(
            functools.partial(
                is_high_dim_transpose_node, graph_converter=self.graph.graph, max_transpose_dims=self.max_transpose_dims
            )
        )

        remove_ids = []
        ops = []
        restore_mapping = []
        for trans in vertices:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in trans.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in trans['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            remove_ids.append(trans.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(vertices, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for trans, mapping in zip(sorted_ops, restore_mapping):
            input_tensor = trans.inputs[0]
            perm_tensor = trans.inputs[1]
            output_tensor = trans.outputs[0]

            input_shape = input_tensor.shape
            perm = perm_tensor.tensor
            output_shape = output_tensor.shape

            last_perm = None
            last_dim = None
            cum_dim = None
            new_shape = []
            new_perm = []
            for d, p in zip(input_shape, perm):
                if last_dim is None and last_perm is None:
                    cum_dim = d
                else:
                    if p - last_perm == 1 or d == 1 or cum_dim == 1:
                        cum_dim *= d
                    else:
                        new_shape.append(cum_dim)
                        new_perm.append(last_perm)
                        cum_dim = d

                last_dim = d
                last_perm = p

            new_shape.append(cum_dim)
            new_perm.append(last_perm)

            new_perm_arr = np.argsort(new_perm).astype('int32')

            assert (
                len(new_shape) <= self.max_transpose_dims
            ), f"Don't know how to reduce the number of dims of transpose with input shape {input_shape}, perm {perm}"

            ops = []

            input_reduced = self.create_transform_tensor(
                np.reshape(input_tensor.tensor, new_shape), quantization=input_tensor.quantization
            )
            reduced_shape = self.create_attr_tensor(np.array(new_shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([input_tensor, reduced_shape], [input_reduced], new_shape))

            transposed = self.create_transform_tensor(
                np.transpose(input_reduced.tensor, new_perm_arr), quantization=input_tensor.quantization
            )
            new_perm_tensor = self.create_attr_tensor(np.array(new_perm_arr, dtype='int32'))
            ops.append(tfl.TransposeOperator([input_reduced, new_perm_tensor], [transposed]))

            output_shape_tensor = self.create_attr_tensor(np.array(output_shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([transposed, output_shape_tensor], [output_tensor], output_shape))

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    @class_conditional(lambda self: self.group_conv_rewrite)
    def group_conv_rewrite_pass(self):
        vertices = self.graph.graph.vs.select(functools.partial(is_group_conv_node, graph_converter=self.graph.graph))

        remove_ids = []
        ops = []
        restore_mapping = []
        for conv in vertices:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in conv.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in conv['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            remove_ids.append(conv.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(vertices, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for conv, mapping in zip(sorted_ops, restore_mapping):
            input_tensor = conv.inputs[0]
            weight_tensor = conv.inputs[1]
            bias_tensor = conv.inputs[2] if len(conv.inputs) > 2 else None
            output_tensor = conv.outputs[0]

            num_input_channel = input_tensor.shape[3]
            num_weight_channel = weight_tensor.shape[3]
            num_chunks = num_input_channel // num_weight_channel

            ops = []

            input_tensors = [
                self.create_transform_tensor(arr, quantization=input_tensor.quantization)
                for arr in np.split(input_tensor.tensor, num_chunks, 3)
            ]
            output_tensors = [
                self.create_transform_tensor(arr, quantization=output_tensor.quantization)
                for arr in np.split(output_tensor.tensor, num_chunks, 3)
            ]
            weights = [
                self.create_attr_tensor(arr, quantization=weight_tensor.quantization)
                for arr in np.split(weight_tensor.tensor, num_chunks, 0)
            ]

            if bias_tensor is not None:
                biases = [
                    self.create_attr_tensor(arr, quantization=bias_tensor.quantization)
                    for arr in np.split(bias_tensor.tensor, num_chunks, 0)
                ]
            else:
                biases = [None] * num_chunks

            dim_tensor = self.create_attr_tensor(np.array([3], dtype='int32'))
            ops.append(tfl.SplitOperator([dim_tensor, input_tensor], input_tensors, num_chunks))

            for it, ot, w, b in zip(input_tensors, output_tensors, weights, biases):
                inputs = [it, w]
                if b is not None:
                    inputs.append(b)
                ops.append(
                    tfl.Conv2dOperator(
                        inputs,
                        [ot],
                        strideH=conv.strideH,
                        strideW=conv.strideW,
                        dilationHFactor=conv.dilationHFactor,
                        dilationWFactor=conv.dilationWFactor,
                        fusedActivationFunction=conv.fusedActivationFunction,
                        padding=conv.padding,
                    )
                )

            ops.append(tfl.ConcatenationOperator(output_tensors, [output_tensor], 3))

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    @class_conditional(lambda self: self.group_conv_rewrite)
    def group_deconv_rewrite_pass(self):
        vertices = self.graph.graph.vs.select(functools.partial(is_group_deconv_node, graph_converter=self.graph.graph))

        remove_ids = []
        ops = []
        restore_mapping = []
        for conv in vertices:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in conv.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in conv['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            remove_ids.append(conv.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(vertices, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for conv, mapping in zip(sorted_ops, restore_mapping):
            input_tensor = conv.inputs[2]
            weight_tensor = conv.inputs[1]
            output_shape_tensor = conv.inputs[0]
            bias_tensor = conv.inputs[3] if len(conv.inputs) > 3 else None
            output_tensor = conv.outputs[0]

            num_output_channel = output_tensor.shape[3]
            num_weight_channel = weight_tensor.shape[0]
            num_chunks = num_output_channel // num_weight_channel

            ops = []

            input_tensors = [
                self.create_transform_tensor(arr, quantization=input_tensor.quantization)
                for arr in np.split(input_tensor.tensor, num_chunks, 3)
            ]
            output_tensors = [
                self.create_transform_tensor(arr, quantization=output_tensor.quantization)
                for arr in np.split(output_tensor.tensor, num_chunks, 3)
            ]
            weights = [
                self.create_attr_tensor(arr, quantization=weight_tensor.quantization)
                for arr in np.split(weight_tensor.tensor, num_chunks, 3)
            ]

            if bias_tensor is not None:
                biases = [
                    self.create_attr_tensor(arr, quantization=bias_tensor.quantization)
                    for arr in np.split(bias_tensor.tensor, num_chunks, 0)
                ]
            else:
                biases = [None] * num_chunks

            new_os = output_shape_tensor.tensor.copy()
            new_os[3] = num_weight_channel
            new_ost = self.create_attr_tensor(new_os)
            dim_tensor = self.create_attr_tensor(np.array([3], dtype='int32'))
            ops.append(tfl.SplitOperator([dim_tensor, input_tensor], input_tensors, num_chunks))

            for it, ot, w, b in zip(input_tensors, output_tensors, weights, biases):
                inputs = [new_ost, w, it]
                if b is not None:
                    inputs.append(b)
                ops.append(
                    tfl.TransposeConvOperator(
                        inputs,
                        [ot],
                        padding=conv.padding,
                        strideH=conv.strideH,
                        strideW=conv.strideW,
                    )
                )

            ops.append(tfl.ConcatenationOperator(output_tensors, [output_tensor], 3))

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    @class_conditional(lambda self: self.tflite_micro_rewrite)
    def cat_split_pass(self):
        vertices = self.graph.graph.vs.select(functools.partial(is_large_cat_node, graph_converter=self.graph.graph))

        remove_ids = []
        ops = []
        restore_mapping = []
        for cat in vertices:
            restore_nodes = []
            # For each node that is next of a transformable node,
            #  a. if it is an output node, remove it anyway since it will always be reconstructed
            #  b. otherwise, record the info of the edge so that we may restore it after reconstruction
            for out_edge in cat.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_ids.append(next_node.index)
                    del self.graph.tensor_map[next_node['outputs'][0]]
                    del self.graph.tensor_node_map[next_node['outputs'][0]]
                else:
                    restore_nodes.append((out_edge['name'], next_node['name']))

            # Remove the mapping since they are going to be removed
            for output_name in cat['outputs']:
                del self.graph.tensor_map[output_name]
                del self.graph.tensor_node_map[output_name]

            restore_mapping.append(restore_nodes)
            remove_ids.append(cat.index)

        # Make sure the nodes are topologically sorted
        sorted_ops = [node['op'] for node in sorted(vertices, key=lambda x: int(re.search(r'\d+', x['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for cat, mapping in zip(sorted_ops, restore_mapping):
            input_tensors = cat.inputs
            layer_inputs = input_tensors
            output_tensor = cat.outputs[0]

            axis = cat.axis
            last_layer = False

            ops = []

            while True:
                layer_outputs = []

                while len(layer_inputs) > 0:
                    curr_inputs = layer_inputs[:10]

                    input_arrs = [t.tensor for t in curr_inputs]
                    output_arr = np.concatenate(input_arrs, axis)

                    if last_layer:
                        curr_output = output_tensor
                    else:
                        curr_output = self.create_transform_tensor(output_arr, quantization=output_tensor.quantization)
                        layer_outputs.append(curr_output)

                    ops.append(tfl.ConcatenationOperator(curr_inputs, [curr_output], axis))

                    layer_inputs = layer_inputs[10:]

                if len(layer_outputs) == 0:
                    break
                elif len(layer_outputs) <= 10:
                    last_layer = True

                layer_inputs = layer_outputs

            for op in ops:
                self.graph.add_operator(op, transform=True)

            self.graph.try_restore_edges(mapping)

    def input_transpose_pass(self):
        nhwc2nchw_perm = np.array([0, 3, 1, 2], dtype='int32')
        nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')

        remove_edges = []
        for name, transpose in zip(self.graph.inputs, self.graph.input_transpose):
            if transpose is True:
                node_name = self.graph.tensor_node_map[name]
                node = self.graph.graph.vs.find(name=node_name)
                assert node['node_type'] == ExtendedOperator.INPUT_NODE

                # For quantized graphs, we insert the transpose op after the quantize op
                next_node = None
                if node.outdegree() == 1:
                    next_node = node.out_edges()[0].target_vertex
                    if next_node['node_type'] != ExtendedOperator.QUANTIZE:
                        next_node = None

                # Transpose input tensor shapes
                input_tensor = self.graph.tensor_map[node['name']]
                input_tensor.tensor = np.transpose(input_tensor.tensor, nchw2nhwc_perm)
                input_tensor.shape = input_tensor.tensor.shape

                # Transpose quantize output tensor shapes
                last_tensor = input_tensor
                last_node = node
                if next_node is not None:
                    last_node = next_node
                    last_tensor = next_node['op'].outputs[0]
                    last_tensor.tensor = np.transpose(last_tensor.tensor, nchw2nhwc_perm)
                    last_tensor.shape = last_tensor.tensor.shape

                # Create new transpose op
                nhwc2nchw_perm_tensor = self.create_attr_tensor(nhwc2nchw_perm)
                transposed = self.create_transform_tensor(
                    np.transpose(last_tensor.tensor, nhwc2nchw_perm), quantization=last_tensor.quantization
                )
                transpose_op = tfl.TransposeOperator([last_tensor, nhwc2nchw_perm_tensor], [transposed])
                transpose_op.extra_hints['direction'] = 'down'
                self.graph.add_operator(transpose_op)

                # Get the newly-generated node
                new_node_name = self.graph.tensor_node_map[transposed.name]
                new_node = self.graph.graph.vs.find(name=new_node_name)

                # Connect the transpose op to the graph
                self.graph.replace_next_tensors(last_node, new_node, transposed.name, [new_node_name])

                # Collect the unused connections
                for edge in last_node.out_edges():
                    target_vertex = edge.target_vertex
                    if target_vertex['name'] != new_node_name:
                        remove_edges.append(edge.index)

        # Remove the collected edges
        self.graph.graph.delete_edges(remove_edges)

    @class_conditional(lambda self: self.quantize_input_output_type is not None)
    def quantize_input_output_type_pass(self):
        remove_edges = []
        remove_vertices = []
        for i, name in enumerate(self.graph.inputs):
            if self.fuse_input_indices is not None:
                if i not in self.fuse_input_indices:
                    continue

            node_name = self.graph.tensor_node_map[name]
            node = self.graph.graph.vs.find(name=node_name)
            assert node['node_type'] == ExtendedOperator.INPUT_NODE

            # Update input tensor
            input_tensor = self.graph.tensor_map[node['outputs'][0]]
            input_type = str(input_tensor.dtype)
            if input_type == self.quantize_input_output_type:
                continue

            input_arr = input_tensor.tensor.copy()
            input_quantization = copy.deepcopy(input_tensor.quantization)
            if input_type == 'int8' and self.quantize_input_output_type == 'uint8':
                input_tensor.tensor = (input_tensor.tensor.astype('int32') + 128).astype('uint8')
                input_tensor.quantization.zero_point += 128
                input_tensor.dtype = input_tensor.tensor.dtype
            elif input_type == 'uint8' and self.quantize_input_output_type == 'int8':
                input_tensor.tensor = (input_tensor.tensor.astype('int32') - 128).astype('int8')
                input_tensor.quantization.zero_point -= 128
                input_tensor.dtype = input_tensor.tensor.dtype
            else:
                raise AssertionError(
                    f'Unsupported types: input_type: {input_type}, quantize_input_type:'
                    f' {self.quantize_input_output_type}'
                )

            # Create new quantize op
            requantized = self.create_transform_tensor(input_arr, quantization=input_quantization)
            quantize_op = tfl.QuantizeOperator([input_tensor], [requantized])
            self.graph.add_operator(quantize_op)

            # Get the newly-generated node
            new_node_name = self.graph.tensor_node_map[requantized.name]
            new_node = self.graph.graph.vs.find(name=new_node_name)

            # Connect the quantize op to the graph
            self.graph.replace_next_tensors(node, new_node, requantized.name, [new_node_name])

            # Collect the unused connections
            for edge in node.out_edges():
                target_vertex = edge.target_vertex
                if target_vertex['name'] != new_node_name:
                    remove_edges.append(edge.index)

        output_mapping = {}
        for i, name in enumerate(self.graph.outputs):
            if self.fuse_output_indices is not None:
                if i not in self.fuse_output_indices:
                    continue

            output_tensor = self.graph.tensor_map[name]
            output_type = str(output_tensor.dtype)
            if output_type == self.quantize_input_output_type:
                continue

            node_name = self.graph.tensor_node_map[name]
            node = self.graph.graph.vs.find(name=node_name)

            for edge in node.out_edges():
                next_node = edge.target_vertex

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    remove_vertices.append(next_node.index)

            # Update output tensor
            output_arr = output_tensor.tensor.copy()
            output_quantization = copy.deepcopy(output_tensor.quantization)
            if output_type == 'int8' and self.quantize_input_output_type == 'uint8':
                output_arr = (output_arr.astype('int32') + 128).astype('uint8')
                output_quantization.zero_point += 128
            elif output_type == 'uint8' and self.quantize_input_output_type == 'int8':
                output_arr = (output_arr.astype('int32') - 128).astype('int8')
                output_quantization.zero_point -= 128
            else:
                raise AssertionError(
                    f'Unsupported types: output_type: {output_type}, quantize_input_type:'
                    f' {self.quantize_input_output_type}'
                )

            requantized = self.create_transform_tensor(output_arr, quantization=output_quantization)
            quantize_op = tfl.QuantizeOperator([output_tensor], [requantized])
            self.graph.add_operator(quantize_op)

            output_mapping[name] = requantized.name

        if len(output_mapping) > 0:
            new_outputs = []
            output_names = []
            for name in self.graph.outputs:
                if name in output_mapping:
                    new_outputs.append(output_mapping[name])
                    output_names.append(output_mapping[name])
                else:
                    new_outputs.append(name)

            self.graph.outputs.clear()
            self.graph.outputs.extend(new_outputs)
            self.graph.add_outputs(output_names)

        # Remove the collected edges & vertices
        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

    def output_transpose_pass(self):
        nhwc2nchw_perm = np.array([0, 3, 1, 2], dtype='int32')
        nchw2nhwc_perm = np.array([0, 2, 3, 1], dtype='int32')

        if isinstance(self.graph.output_transpose, (list, tuple)):
            assert len(self.graph.output_transpose) == len(self.graph.outputs)
        else:
            self.graph.output_transpose = [self.graph.output_transpose] * len(self.graph.outputs)

        filtered_dict = {}
        for i, (name, transpose) in enumerate(zip(self.graph.outputs, self.graph.output_transpose)):
            if name in filtered_dict:
                old_transpose = filtered_dict[name]
                assert (
                    transpose == old_transpose
                ), f"outputs {i} points to an exising tensor {name}, but their property `output_transpose` is different"
            else:
                filtered_dict[name] = transpose

        prev_modify_node_indices = {}
        prev_modify_next_indices = {}
        next_modify_node_indices = {}
        for name, transpose in filtered_dict.items():
            if name in self.graph.tensor_map:
                tensor = self.graph.tensor_map[name]
                if transpose is None:
                    transpose = len(tensor.shape) == 4
            else:
                transpose = False

            for i, n in enumerate(self.graph.outputs):
                if name == n:
                    self.graph.output_transpose[i] = transpose

            if transpose:
                node_name = self.graph.tensor_node_map[name]
                node = self.graph.graph.vs.find(name=node_name)
                tensor_idx = node['outputs'].index(name)

                prev_node = None
                if node['node_type'] == ExtendedOperator.DEQUANTIZE:
                    prev_node_name = self.graph.tensor_node_map[node['op'].inputs[0].name]
                    prev_node = self.graph.graph.vs.find(name=prev_node_name)

                if prev_node is None:
                    next_modify_node_indices.setdefault(node, set())
                    next_modify_node_indices[node].add(tensor_idx)
                else:
                    prev_modify_node_indices.setdefault(node, set())
                    prev_modify_node_indices[node].add(0)
                    prev_modify_next_indices.setdefault(node, set())
                    prev_modify_next_indices[node].add(tensor_idx)

        remove_edges = []
        remove_vertices = []
        actions = []
        for node, index in prev_modify_node_indices.items():
            next_indices = prev_modify_next_indices[node]
            op = node['op']
            tensor_names = [node['outputs'][i] for i in index]

            next_nodes = {}
            for edge in node.out_edges():
                if edge['label'] not in tensor_names:
                    continue

                if edge.index in remove_edges:
                    continue

                tensor_idx = tensor_names.index(edge['label'])
                next_node = self.graph.graph.vs[edge.target]

                if next_node['node_type'] not in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE):
                    next_nodes.setdefault(tensor_idx, [])
                    next_nodes[tensor_idx].append(next_node)

            prev_nodes = []
            prev_output_indices = []
            for i in index:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

            tensor_node_dict = {}
            for prev_node, prev_idx, next_idx in zip(prev_nodes, index, prev_output_indices):
                if prev_node['op'] is None:
                    prev_out = self.graph.tensor_map[prev_node['outputs'][0]]
                else:
                    prev_out = prev_node['op'].outputs[next_idx]
                if prev_out.name in tensor_node_dict:
                    prev_new_out, skip = tensor_node_dict[prev_out.name]
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True, skip)))
                    skip += 1
                    tensor_node_dict[prev_out.name] = (prev_new_out, skip)
                else:
                    perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)
                    prev_new_out = self.create_transform_tensor(
                        np.transpose(prev_out.tensor, nchw2nhwc_perm), quantization=prev_out.quantization
                    )
                    tensor_node_dict[prev_out.name] = (prev_new_out, 1)
                    prev_transpose_op = tfl.TransposeOperator([prev_out, perm_tensor], [prev_new_out])
                    prev_transpose_op.extra_hints['direction'] = 'up'
                    self.graph.add_operator(prev_transpose_op)
                    actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

            tensor_mapping = {}
            for i in next_indices:
                t = op.outputs[i]
                t.tensor = np.transpose(t.tensor, nchw2nhwc_perm)
                t.shape = t.tensor.shape

                if i in next_nodes:
                    new_t = self.create_transform_tensor(np.transpose(t.tensor, nhwc2nchw_perm))
                    perm_t = self.create_attr_tensor(nhwc2nchw_perm)

                    next_transpose_op = tfl.TransposeOperator([t, perm_t], [new_t])
                    next_transpose_op.extra_hints['direction'] = 'down'
                    self.graph.add_operator(next_transpose_op)

                    tensor_mapping[t.name] = new_t

            for nodes in next_nodes.values():
                for n in nodes:
                    next_op = n['op']
                    for i, t in enumerate(next_op.inputs):
                        if t.name in tensor_mapping:
                            actions.append((self.graph.replace_operator_input, (n, i, tensor_mapping[t.name])))

        for node, index in next_modify_node_indices.items():
            op = node['op']
            tensor_names = [node['outputs'][i] for i in index]
            out_nodes = []
            next_nodes = []
            next_edges = []
            for edge in node.out_edges():
                if edge['label'] not in tensor_names:
                    continue

                if edge.index in remove_edges:
                    continue

                next_node = self.graph.graph.vs[edge.target]
                tensor_idx = tensor_names.index(edge['label'])

                if next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                    out_nodes.append(next_node)
                elif next_node['node_type'] != ExtendedOperator.UNUSED_NODE:
                    next_nodes.append(next_node)
                    next_edges.append(edge)

            remove_vertices.extend([x.index for x in out_nodes])
            remove_edges.extend([x.index for x in next_edges])

            for n in out_nodes:
                del self.graph.tensor_map[n['outputs'][0]]
                del self.graph.tensor_node_map[n['outputs'][0]]

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                if i not in index:
                    continue

                op_out.tensor = np.transpose(op_out.tensor, nchw2nhwc_perm)
                op_out.shape = op_out.tensor.shape

                perm_tensor = self.create_attr_tensor(nchw2nhwc_perm)
                new_out = self.create_transform_tensor(
                    np.transpose(op_out.tensor, nhwc2nchw_perm), quantization=op_out.quantization
                )

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                next_transpose_op = tfl.TransposeOperator([new_out, perm_tensor], [op_out])
                next_transpose_op.extra_hints['direction'] = 'up'
                self.graph.add_operator(next_transpose_op)

                tensor_node_dict[op_out.name] = (
                    self.graph.graph.vs.find(name=self.graph.tensor_node_map[new_out.name]),
                    new_out.name,
                )

            # Connect next edges and replace next tensors
            for edge in next_edges:
                old_name = edge['name']
                source, new_name = tensor_node_dict[old_name]
                target = edge.target_vertex
                self.graph.graph.add_edge(source, target, name=new_name, label=new_name)

                op = target['op']
                for i, op_input in enumerate(op.inputs):
                    if op_input.name == old_name:
                        op.inputs[i] = self.graph.tensor_map[new_name]
                        break

        # Process actions
        ids = []
        for func, args in actions:
            node = args[0]
            res = func(*args)
            if res is not None:
                ids.extend(res)

        remove_edges = list(set(remove_edges + ids))

        self.graph.graph.delete_edges(remove_edges)
        self.graph.graph.delete_vertices(remove_vertices)

    def connect_unused_tensors_pass(self):
        filtered_nodes = self.graph.graph.vs.select(
            functools.partial(is_multi_output_op_node, graph_converter=self.graph.graph)
        )

        list_unpack_names = set([i for s in self.graph.iterable_map.values() for i in s])
        all_tensors = set(self.graph.graph.es['label'])
        names = []
        for node in filtered_nodes:
            output_names = node['outputs']

            # Recognizes the pattern SPLIT -> (RESHAPE, ..., RESHAPE)
            if not list_unpack_names.isdisjoint(set(output_names)):
                output_names = []
                outdegree = 0
                for edge in node.out_edges():
                    target_vertex = edge.target_vertex
                    if target_vertex['node_type'] == ExtendedOperator.RESHAPE:
                        outdegree += target_vertex.outdegree()
                        output_names.append(target_vertex['outputs'][0])

                    # Only nodes with partially unused tensors are supported
                    if outdegree == 0:
                        continue

            for out in output_names:
                if out not in all_tensors:
                    names.append(out)

        self.graph.add_outputs(names, ExtendedOperator.UNUSED_NODE)

    def output_list_unpack_pass(self):
        output_names = []
        unpacked_outputs = []
        for name in self.graph.outputs:
            if name in self.graph.iterable_map:
                names = self.graph.get_list_expanded_names(name)
                unpacked_outputs.extend(names)
                output_names.extend(names)
            else:
                unpacked_outputs.append(name)

        self.graph.outputs.clear()
        self.graph.outputs.extend(unpacked_outputs)
        self.graph.add_outputs(output_names)

    @class_conditional(lambda self: self.fuse_quant)
    def fuse_quant_dequant_nodes(self):
        edges = self.graph.graph.es.select(functools.partial(is_quant_dequant_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        remove_vertices = []
        input_mapping = {}
        output_mapping = {}
        for prev, next in filtered_pairs:
            if prev['node_type'] == ExtendedOperator.INPUT_NODE:
                input_name = prev['outputs'][0]
                if self.fuse_input_indices is not None:
                    input_idx = self.graph.inputs.index(input_name)
                    if input_idx not in self.fuse_input_indices:
                        continue
                remove_vertices.append(prev)
                next['node_type'] = prev['node_type']
                next['op'] = None
                input_mapping.setdefault(input_name, [])
                input_mapping[input_name].extend(next['outputs'])
            else:
                if prev['op'] is not None:
                    prev_name = prev['op'].outputs[0].name
                    if self.fuse_output_indices is not None:
                        output_idx = self.graph.outputs.index(prev_name)
                        if output_idx not in self.fuse_output_indices:
                            continue
                    prev['node_type'] = next['node_type']
                    new_name = prev['op'].inputs[0].name
                    prev['op'] = None
                    output_mapping.setdefault(prev_name, [])
                    output_mapping[prev_name].append(new_name)
                remove_vertices.append(next)

        self.graph.graph.delete_vertices(remove_vertices)

        if len(input_mapping) > 0:
            new_inputs = []
            for name in self.graph.inputs:
                if name in input_mapping:
                    new_inputs.extend(input_mapping[name])
                else:
                    new_inputs.append(name)

            self.graph.inputs.clear()
            self.graph.inputs.extend(new_inputs)

        if len(output_mapping) > 0:
            new_outputs = []
            for name in self.graph.outputs:
                if name in output_mapping:
                    new_outputs.extend(output_mapping[name])
                else:
                    new_outputs.append(name)

            self.graph.outputs.clear()
            self.graph.outputs.extend(new_outputs)

    def optimize(self):
        # Input/output passes
        self.output_list_unpack_pass()
        self.input_transpose_pass()
        self.output_transpose_pass()

        # Connect unused tensors with special nodes
        self.connect_unused_tensors_pass()

        # Transpose, Reshape and NO-OP cleanup
        self.branch_reshape_expand_pass()
        self.fuse_simple_reshape_pass()
        self.branch_transpose_expand_pass()
        self.fuse_simple_transpose_pass()
        for branch in (False, True):
            self.remove_noop_pass(branch)
        self.fuse_wrapped_reshape_within_transpose_pass()

        # Buffer folding, which is needed by the fusion passes below
        for _ in range(2):
            self.fold_reshape_buffer()
            self.fold_transpose_buffer()

        # Move `transpose` ops for the rewrite quantizable pass
        self.elementwise_op_transpose_passthrough_pass(quantizable_ops_only=True)
        self.branch_transpose_expand_pass()
        self.fuse_simple_transpose_pass()

        # Fuse reciprocal and sqrt
        self.fuse_reciprocal_sqrt()

        # Map quantizable ops to quantized kernels
        self.elementwise_op_quantize_passthrough_pass()

        # Remove consecutive dequantize and quantize nodes
        self.fuse_dequant_quant_pass(q_first=False)

        # OP fusion passes before transformation
        self.fuse_conv_fc_bn()
        self.fuse_activation()
        self.fuse_requantize()

        # Convert TinyNeuralNetwork ops to TFLite ops
        self.transform_graph()

        # OP fusion passes after transformation
        self.fuse_bmm_add_pass()
        self.fuse_activation()

        # Transpose and reshape cleanup
        self.branch_reshape_expand_pass()
        self.branch_transpose_expand_pass()
        self.fuse_simple_transpose_pass()
        self.fuse_simple_reshape_pass()

        # Branch transpose & reshape cleanup
        for i in range(11):
            t_count = self.elementwise_op_transpose_passthrough_pass()
            self.branch_transpose_expand_pass()
            self.fuse_simple_transpose_pass()

            r_count = self.elementwise_op_reshape_passthrough_pass()
            self.branch_reshape_expand_pass()
            self.fuse_simple_reshape_pass()

            c_count = self.elementwise_reshape_transpose_passthrough_pass()
            self.branch_transpose_expand_pass()
            self.fuse_simple_transpose_pass()

            if t_count + r_count + c_count == 0:
                log.debug(f'elem p/t pass finished in {i + 1} steps')
                break

        # Other cleanups
        self.fuse_simple_slice_pass()
        for branch in (False, True):
            self.remove_noop_pass(branch)
        self.fuse_wrapped_reshape_within_transpose_pass()

        # Buffer folding
        for _ in range(2):
            self.fold_reshape_buffer()
            self.fold_transpose_buffer()

        # Transpose and reshape cleanup
        for _ in range(2):
            self.transpose_to_reshape_pass()
            self.fuse_simple_reshape_pass()
            self.fuse_simple_transpose_pass()

        self.lower_transpose_dim_pass()

        # Some advanced fusion logic
        self.fuse_conv2d_gather()

        # Remove consecutive dequantize and quantize nodes
        self.fuse_dequant_quant_pass(q_first=True)

        # Fuse reciprocal and sqrt
        self.fuse_reciprocal_sqrt()

        # Fuse activation
        self.fuse_activation()

        # Fuse quant/dequant nodes
        self.fuse_quant_dequant_nodes()

        # Input output quantize type
        self.quantize_input_output_type_pass()

        # Fuse same padding
        self.fuse_same_padding()
        self.fuse_same_padding_slicing()

        # Group conv & deconv
        self.group_conv_rewrite_pass()
        self.group_deconv_rewrite_pass()

        # TFLite micro specific
        self.cat_split_pass()
        self.split_requantize()

        # Final cleanup
        self.cleanup_dead_nodes()


def is_bn_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type']
        in (ExtendedOperator.GENERIC_CONV, ExtendedOperator.GENERIC_DECONV, ExtendedOperator.FULLY_CONNECTED)
        and target_vertex['node_type'] == ExtendedOperator.BATCH_NORM
        and source_vertex.outdegree() == 1
        and target_vertex['op'].inputs[1].buffer is not None
        and target_vertex['op'].inputs[2].buffer is not None
        and source_vertex['op'].inputs[1].buffer is not None
        and (
            target_vertex['op'].fusedActivationFunction == ActivationFunctionType.NONE
            or source_vertex['op'].fusedActivationFunction
            in (ActivationFunctionType.NONE, target_vertex['op'].fusedActivationFunction)
        )
    )


def is_padding_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] in (ExtendedOperator.PAD, ExtendedOperator.PADV2)
        and (
            len(source_vertex['op'].inputs) == 2
            or (
                len(source_vertex['op'].inputs) == 3
                and source_vertex['op'].inputs[2].dtype == np.dtype('float32')
                and (
                    (
                        source_vertex['op'].inputs[2].tensor[0] == 0.0
                        and target_vertex['node_type'] != ExtendedOperator.MAX_POOL_2D
                    )
                    or (
                        source_vertex['op'].inputs[2].tensor[0] == np.finfo(np.float32).min
                        and target_vertex['node_type'] == ExtendedOperator.MAX_POOL_2D
                    )
                )
            )
        )
        and target_vertex['node_type']
        in (
            ExtendedOperator.CONV_2D,
            ExtendedOperator.CONV_3D,
            ExtendedOperator.DEPTHWISE_CONV_2D,
            ExtendedOperator.MAX_POOL_2D,
        )
        and source_vertex.outdegree() == 1
        and target_vertex['op'].padding == Padding.VALID
    )


def is_slicing_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        target_vertex['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE)
        and (
            len(target_vertex['op'].inputs) == 3
            or (len(target_vertex['op'].inputs) == 4 and np.all(target_vertex['op'].inputs[3].tensor == 1))
        )
        and source_vertex['node_type']
        in (
            ExtendedOperator.TRANSPOSE_CONV,
            ExtendedOperator.CONV_3D_TRANSPOSE,
        )
        and source_vertex.outdegree() == 1
        and source_vertex['op'].padding == Padding.VALID
    )


def is_requantize_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type']
        in (
            ExtendedOperator.FULLY_CONNECTED,
            ExtendedOperator.GENERIC_CONV,
            ExtendedOperator.ADD,
            ExtendedOperator.SUB,
            ExtendedOperator.MUL,
            ExtendedOperator.DIV,
            ExtendedOperator.MAX_POOL_2D,
            ExtendedOperator.AVERAGE_POOL_2D,
        )
        and source_vertex['op'].outputs[0].quantization is not None
        and target_vertex['node_type'] == ExtendedOperator.QUANTIZE
    )


def is_activ_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type']
        in (
            ExtendedOperator.FULLY_CONNECTED,
            ExtendedOperator.GENERIC_CONV,
            ExtendedOperator.ADD,
            ExtendedOperator.SUB,
            ExtendedOperator.MUL,
            ExtendedOperator.DIV,
            ExtendedOperator.MAX_POOL_2D,
            ExtendedOperator.AVERAGE_POOL_2D,
        )
        and target_vertex['node_type'] in (ExtendedOperator.RELU, ExtendedOperator.RELU6)
        and source_vertex['op'].fusedActivationFunction == ActivationFunctionType.NONE
        and source_vertex.outdegree() == 1
    )


def is_requantize_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return (
        vertex['node_type'] == ExtendedOperator.QUANTIZE
        and vertex['op'].inputs[0].quantization is not None
        and vertex['op'].outputs[0].quantization is not None
    )


def is_large_cat_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return vertex['node_type'] == ExtendedOperator.CONCATENATION and len(vertex['op'].inputs) > 10


def is_high_dim_transpose_node(vertex: ig.Vertex, graph_converter: ig.Graph, max_transpose_dims: int):
    return vertex['node_type'] == ExtendedOperator.TRANSPOSE and vertex['op'].inputs[1].tensor.size > max_transpose_dims


def is_group_conv_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return (
        vertex['node_type'] == ExtendedOperator.CONV_2D
        and vertex['op'].inputs[0].shape[3] != vertex['op'].inputs[1].shape[3]
    )


def is_group_deconv_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return (
        vertex['node_type'] == ExtendedOperator.TRANSPOSE_CONV
        and vertex['op'].outputs[0].shape[3] != vertex['op'].inputs[1].shape[0]
    )


def is_transformable_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return vertex['node_type'] <= ExtendedOperator.BATCH_NORM and vertex.outdegree() >= 1


def is_transformable_transpose_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return (
        vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and vertex.outdegree() >= 1
        and is_transpose_same_to_reshape_op(vertex['op'])
    )


def is_multi_output_op_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return vertex['node_type'] >= 0 and len(vertex['outputs']) > 1 and vertex.outdegree() > 0


def is_quantize_elementwise_op_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        (
            source_vertex['node_type'] == ExtendedOperator.DEQUANTIZE
            and is_quantizable_rewrite_op(target_vertex['node_type'], target_vertex['op'])
        )
        or (
            target_vertex['node_type'] == ExtendedOperator.QUANTIZE
            and is_quantizable_rewrite_op(source_vertex['node_type'], source_vertex['op'])
        )
    ) and target_vertex['op'].inputs[0].name in source_vertex['outputs']


def is_transpose_reshape_op_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        (
            source_vertex['node_type'] == ExtendedOperator.TRANSPOSE
            and target_vertex['node_type'] == ExtendedOperator.RESHAPE
        )
        or (
            target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
            and source_vertex['node_type'] == ExtendedOperator.RESHAPE
        )
    ) and target_vertex['op'].inputs[0].name in source_vertex['outputs']


def is_transpose_elementwise_op_edge(edge: ig.Edge, graph_converter: ig.Graph, quantizable_ops_only: bool):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    if quantizable_ops_only:
        is_unary = is_elementwise_unary_quantizable_op
        is_binary = is_elementwise_binary_quantizable_op
    else:
        is_unary = is_elementwise_unary_op
        is_binary = is_elementwise_binary_op

    return (
        (
            source_vertex['node_type'] == ExtendedOperator.TRANSPOSE
            and (
                is_unary(target_vertex['node_type'], target_vertex['op'])
                or is_binary(target_vertex['node_type'], target_vertex['op'])
            )
        )
        or (
            target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
            and (
                is_unary(source_vertex['node_type'], source_vertex['op'])
                or is_binary(source_vertex['node_type'], source_vertex['op'])
            )
        )
    ) and (
        (
            target_vertex['node_type'] != ExtendedOperator.SPLIT
            and target_vertex['op'].inputs[0].name in source_vertex['outputs']
        )
        or (
            target_vertex['node_type'] == ExtendedOperator.SPLIT
            and target_vertex['op'].inputs[1].name in source_vertex['outputs']
        )
    )


def is_reshape_elementwise_op_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        (
            source_vertex['node_type'] == ExtendedOperator.RESHAPE
            and (
                is_elementwise_unary_op(target_vertex['node_type'], target_vertex['op'])
                or is_elementwise_binary_op(target_vertex['node_type'], target_vertex['op'])
            )
        )
        or (
            target_vertex['node_type'] == ExtendedOperator.RESHAPE
            and (
                is_elementwise_unary_op(source_vertex['node_type'], source_vertex['op'])
                or is_elementwise_binary_op(source_vertex['node_type'], source_vertex['op'])
            )
        )
    ) and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_elementwise_reduce_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return (
        op_code
        in (
            ExtendedOperator.SUM,
            ExtendedOperator.ARG_MIN,
            ExtendedOperator.ARG_MAX,
            ExtendedOperator.REDUCE_MIN,
            ExtendedOperator.REDUCE_MAX,
            ExtendedOperator.REDUCE_PROD,
        )
        and len(op.inputs[0].shape) == len(op.outputs[0].shape)
    ) or (
        op_code == ExtendedOperator.MEAN
        and len(op.inputs[0].shape) == len(op.outputs[0].shape)
        and (
            len(op.inputs[0].shape) != 4
            or (
                not np.array_equal(op.inputs[1].tensor, np.array([1, 2], dtype='int32'))
                and not np.array_equal(op.inputs[1].tensor, np.array([2, 1], dtype='int32'))
            )
        )
    )


def is_elementwise_unary_quantizable_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return op_code in (
        ExtendedOperator.SOFTMAX,
        ExtendedOperator.LOG_SOFTMAX,
    )


def is_elementwise_binary_quantizable_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return False


def is_elementwise_unary_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return op_code in (
        ExtendedOperator.RELU,
        ExtendedOperator.SIN,
        ExtendedOperator.COS,
        ExtendedOperator.TANH,
        ExtendedOperator.ELU,
        ExtendedOperator.PRELU,
        ExtendedOperator.EXP,
        ExtendedOperator.LOG,
        ExtendedOperator.NEG,
        ExtendedOperator.FLOOR,
        ExtendedOperator.RELU6,
        ExtendedOperator.QUANTIZE,
        ExtendedOperator.DEQUANTIZE,
        ExtendedOperator.SQRT,
        ExtendedOperator.RSQRT,
        ExtendedOperator.CAST,
        ExtendedOperator.LOGISTIC,
        ExtendedOperator.HARD_SWISH,
        ExtendedOperator.LEAKY_RELU,
        ExtendedOperator.SPLIT,
        ExtendedOperator.SPLIT_V,
        ExtendedOperator.UNPACK,
        ExtendedOperator.PAD,
        ExtendedOperator.PADV2,
        ExtendedOperator.MIRROR_PAD,
        ExtendedOperator.SLICE,
        ExtendedOperator.STRIDED_SLICE,
        ExtendedOperator.TILE,
        ExtendedOperator.GATHER,
        ExtendedOperator.ABS,
    ) or is_elementwise_reduce_op(op_code, op)


def is_quantizable_rewrite_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return op_code in (
        ExtendedOperator.BATCH_MATMUL,
        ExtendedOperator.SOFTMAX,
        ExtendedOperator.LOG_SOFTMAX,
        ExtendedOperator.ABS,
        ExtendedOperator.SUM,
        ExtendedOperator.DIV,
        ExtendedOperator.RSQRT,
    )


def is_elementwise_binary_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return (
        op_code
        in (
            ExtendedOperator.CONCATENATION,
            ExtendedOperator.PACK,
            ExtendedOperator.ADD,
            ExtendedOperator.SUB,
            ExtendedOperator.MUL,
            ExtendedOperator.DIV,
        )
        and len(op.inputs) >= 2
    )


def is_non_passthrough_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return op_code in (
        ExtendedOperator.CONV_2D,
        ExtendedOperator.AVERAGE_POOL_2D,
        ExtendedOperator.DEPTHWISE_CONV_2D,
        ExtendedOperator.MAX_POOL_2D,
    )


def is_ending_with_noop_edge(edge: ig.Edge, graph_converter: ig.Graph, branch: bool = False):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    if branch:
        source_cond_var = source_vertex.outdegree() >= 1
    else:
        source_cond_var = source_vertex.outdegree() == 1

    return (
        source_cond_var
        and target_vertex.outdegree() >= 1
        and target_vertex['op'] is not None
        and target_vertex['op'].inputs[0].name in source_vertex['outputs']
        and (
            (
                target_vertex['node_type'] == ExtendedOperator.RESHAPE
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
            )
            or (
                target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
                and (np.diff(target_vertex['op'].inputs[1].tensor) == 1).all()
            )
            or (
                target_vertex['node_type']
                in (ExtendedOperator.PAD, ExtendedOperator.PADV2, ExtendedOperator.MIRROR_PAD)
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
            )
            or (
                target_vertex['node_type'] == ExtendedOperator.TILE
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
            )
            or (
                target_vertex['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE)
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
            )
            or (
                target_vertex['node_type'] == ExtendedOperator.CONCATENATION
                and len(target_vertex['op'].inputs) == 1
                and len(target_vertex['op'].outputs) == 1
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
            )
            or (
                target_vertex['node_type'] == ExtendedOperator.GATHER
                and target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape
                and (np.diff(target_vertex['op'].inputs[1].tensor) == 1).all()
            )
            or (
                target_vertex['node_type'] == ExtendedOperator.CAST
                and target_vertex['op'].inDataType == target_vertex['op'].outDataType
            )
        )
    )


def is_bmm_add_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    out_dim_idx = None
    if source_vertex['node_type'] == ExtendedOperator.BATCH_MATMUL:
        out_dim_idx = -1
    elif source_vertex['node_type'] == ExtendedOperator.FULLY_CONNECTED:
        out_dim_idx = 0

    return (
        out_dim_idx is not None
        and target_vertex['node_type'] == ExtendedOperator.ADD
        and source_vertex['op'].inputs[0].tensor.ndim >= 2
        and source_vertex['op'].inputs[1].tensor.ndim == 2
        and target_vertex['op'].inputs[1].tensor.ndim == 1
        and target_vertex['op'].inputs[1].shape[0] == source_vertex['op'].inputs[1].shape[out_dim_idx]
        and source_vertex.outdegree() == 1
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_wrapped_reshape_within_transpose_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        (
            (
                target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
                and source_vertex['node_type'] == ExtendedOperator.RESHAPE
            )
            or (
                source_vertex['node_type'] == ExtendedOperator.TRANSPOSE
                and target_vertex['node_type'] == ExtendedOperator.RESHAPE
            )
        )
        and source_vertex.outdegree() == 1
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_slice_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE)
        and source_vertex.outdegree() == 1
        and target_vertex['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE)
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_transpose_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and source_vertex.outdegree() == 1
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_reshape_branch_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.RESHAPE
        and source_vertex.outdegree() > 1
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_transpose_branch_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and source_vertex.outdegree() > 1
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_dequant_quant_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph, q_first: np.bool):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    if q_first:
        cond = (
            source_vertex['node_type'] == ExtendedOperator.QUANTIZE
            and target_vertex['node_type'] == ExtendedOperator.DEQUANTIZE
        )

    else:
        cond = (
            source_vertex['node_type'] == ExtendedOperator.DEQUANTIZE
            and target_vertex['node_type'] == ExtendedOperator.QUANTIZE
        )

    return (
        cond
        and source_vertex.outdegree() == 1
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_reshape_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.RESHAPE
        and source_vertex.outdegree() == 1
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE
        and target_vertex.outdegree() >= 1
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    )


def is_constant_transpose_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.CONSTANT_NODE
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
        and target_vertex.outdegree() >= 1
    )


def is_constant_reshape_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.CONSTANT_NODE
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
        and target_vertex.outdegree() >= 1
    )


def is_quant_dequant_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return (
        source_vertex['node_type'] == ExtendedOperator.INPUT_NODE
        and target_vertex['node_type'] == ExtendedOperator.QUANTIZE
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name
    ) or (
        source_vertex['node_type'] == ExtendedOperator.DEQUANTIZE
        and target_vertex['node_type'] == ExtendedOperator.OUTPUT_NODE
    )


def is_transpose_same_to_reshape_op(op: tfl.BaseOperator):
    num_elements = np.prod(op.inputs[0].shape)

    input_shape = np.array(op.inputs[0].shape, dtype='int32')
    output_shape = np.array(op.outputs[0].shape, dtype='int32')

    if np.array_equal(input_shape[input_shape != 1], output_shape[output_shape != 1]):
        input_tensor = np.arange(num_elements).reshape(input_shape)
        perm = op.inputs[1].tensor
        new_tensor = np.transpose(input_tensor, perm)

        return np.array_equal(new_tensor.flatten(), input_tensor.flatten())
    else:
        return False


def is_conv2d_gather_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    return (
        source_vertex['node_type'] == ExtendedOperator.CONV_2D
        and target_vertex['node_type'] == ExtendedOperator.GATHER
        and source_vertex.outdegree() == 1
        and target_vertex['op'].inputs[1].buffer is not None
        and target_vertex['op'].axis == 3
        and source_vertex['op'].inputs[1].tensor.shape[0] == target_vertex['op'].inputs[1].tensor.shape[0]
    )


def is_reciprocal_sqrt_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]

    return (
        source_vertex['node_type'] == ExtendedOperator.SQRT
        and target_vertex['node_type'] == ExtendedOperator.DIV
        and source_vertex.outdegree() == 1
    )


def op_input_dims(op: tfl.BaseOperator):
    dim_indices = None

    if isinstance(op, (tfl.ConcatenationOperator, tfl.GatherOperator, tfl.PackOperator, tfl.UnpackOperator)):
        dim_indices = op.axis
    elif isinstance(op, tfl.SplitOperator):
        dim_indices = op.inputs[0].tensor[0]
    elif isinstance(op, tfl.SplitVOperator):
        dim_indices = op.inputs[2].tensor[0]
    elif isinstance(op, (tfl.PadOperator, tfl.Padv2Operator, tfl.MirrorPadOperator)):
        pads = np.sum(op.inputs[1].tensor, axis=-1)
        nonzero_idx = np.nonzero(pads)[0]
        # TODO: support multi indices
        if nonzero_idx.size == 1:
            dim_indices = nonzero_idx[0]
    elif isinstance(op, tfl.PreluOperator):
        w_shape = np.array(op.inputs[1].shape, dtype='int32')
        nonzero_idx = np.nonzero(w_shape != 1)[0]
        if nonzero_idx.size == 1:
            dim_indices = nonzero_idx[0] + 1
    elif isinstance(op, (tfl.SliceOperator, tfl.StridedSliceOperator, tfl.TileOperator)):
        old_shape = np.array(op.inputs[0].shape)
        new_shape = np.array(op.outputs[0].shape)
        diff = new_shape - old_shape
        nonzero_idx = np.nonzero(diff)[0]
        # TODO: support multi indices
        if nonzero_idx.size == 1:
            dim_indices = nonzero_idx[0]
    elif isinstance(
        op,
        (
            tfl.SumOperator,
            tfl.MeanOperator,
            tfl.ArgMinOperator,
            tfl.ArgMaxOperator,
            tfl.ReduceMinOperator,
            tfl.ReduceMaxOperator,
            tfl.ReduceProdOperator,
        ),
    ):
        # TODO: support multi indices
        if op.inputs[1].tensor.size == 1:
            dim_indices = op.inputs[1].tensor[0]
    return dim_indices


def op_input_indices(op: tfl.BaseOperator):
    if isinstance(op, (tfl.ConcatenationOperator, tfl.PackOperator)):
        input_indices = range(len(op.inputs))
    elif isinstance(op, tfl.SplitOperator):
        input_indices = (1,)
    elif isinstance(op, tfl.BatchMatmulOperator):
        input_indices = range(2)
    elif isinstance(op, (tfl.AddOperator, tfl.SubOperator, tfl.MulOperator, tfl.DivOperator)):
        if len(op.inputs[1].shape) == 1 and op.inputs[1].shape[0] == 1:
            input_indices = range(1)
        elif len(op.inputs[0].shape) == 1 and op.inputs[0].shape[0] == 1:
            input_indices = (1,)
        else:
            input_indices = range(2)
    else:
        input_indices = range(1)

    return input_indices


def fuse_bn_weight(eps, scale, var, weight, transpose):
    if transpose:
        shape = [1, -1] + [1] * (len(weight.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(weight.shape) - 2)

    inv = 1 / np.sqrt(var + eps)

    return weight * (scale * inv).reshape(shape)


def fuse_bn_bias(eps, scale, var, mean, bn_b, activ_b):
    inv = 1 / np.sqrt(var + eps)
    if activ_b is not None:
        if activ_b.shape != mean.shape and activ_b.ndim == 1 and activ_b.size == 1:
            activ_b = activ_b.repeat(mean.size)
        return (activ_b - mean) * inv * scale + bn_b
    else:
        return (-mean) * inv * scale + bn_b


def fuse_slices(seq: typing.Iterable[ig.Vertex]):
    cur_start = None
    cur_end = None
    cur_strides = None
    for node in seq:
        assert node['node_type'] in (ExtendedOperator.SLICE, ExtendedOperator.STRIDED_SLICE)
        next_start = node['op'].inputs[1].tensor
        if cur_strides is None:
            cur_strides = np.ones_like(next_start, dtype='int32')
        if cur_start is None:
            cur_start = np.zeros_like(next_start, dtype='int32')
        if node['node_type'] == ExtendedOperator.SLICE:
            next_size = node['op'].inputs[2].tensor
            next_end = cur_start + (next_start + next_size) * cur_strides
            next_strides = np.ones_like(next_start, dtype='int32')
        else:
            next_end = node['op'].inputs[2].tensor
            next_end = cur_start + next_end * cur_strides
            next_strides = node['op'].inputs[3].tensor
        if cur_end is None:
            cur_start = next_start
            cur_end = next_end
            cur_strides = next_strides
        else:
            cur_start += next_start * cur_strides
            cur_end = np.min((cur_end, next_end), axis=0)
            cur_strides = cur_strides * next_strides
    return cur_start, cur_end, cur_strides


def fuse_transpose_perms(seq: typing.Iterable[ig.Vertex]):
    cur_perm = None
    for node in seq:
        assert node['node_type'] == ExtendedOperator.TRANSPOSE
        next_perm = node['op'].inputs[1].tensor
        if cur_perm is None:
            cur_perm = next_perm
        else:
            cur_perm = cur_perm[next_perm]
    return cur_perm


def fuse_transpose_perms_extended(seq: typing.Iterable[ig.Vertex]):
    cur_perm = None
    # Reverse the sequence if dim is expanding
    if seq[1]['node_type'] == ExtendedOperator.RESHAPE:
        if len(seq[1]['op'].inputs[0].shape) < len(seq[1]['op'].outputs[0].shape):
            seq = list(reversed(list(seq)))
    for node in seq:
        if node['node_type'] == ExtendedOperator.TRANSPOSE:
            next_perm = node['op'].inputs[1].tensor
            if cur_perm is None:
                cur_perm = next_perm
            else:
                cur_perm = cur_perm[next_perm]
        elif node['node_type'] == ExtendedOperator.RESHAPE:
            if len(seq[1]['op'].inputs[0].shape) > len(seq[1]['op'].outputs[0].shape):
                old_shape = node['op'].inputs[0].shape
                new_shape = node['op'].outputs[0].shape
            else:
                new_shape = node['op'].inputs[0].shape
                old_shape = node['op'].outputs[0].shape

            if old_shape != new_shape:
                new_shape_padded = list(new_shape) + [None] * (len(old_shape) - len(new_shape))
                next_perm = []
                new_idx = 0
                while new_idx < len(new_shape):
                    for old, item in zip(old_shape, cur_perm):
                        if old == new_shape_padded[new_idx] and item not in next_perm:
                            next_perm.append(item)
                            new_idx += 1
                cur_perm = np.argsort(next_perm)

    return cur_perm


def fuse_connected_edges(
    filtered_pairs: typing.List[typing.Iterable[ig.Vertex]],
) -> typing.List[typing.Iterable[ig.Vertex]]:
    while True:
        heads = {n[0]: i for i, n in enumerate(filtered_pairs)}
        tails = {n[-1]: i for i, n in enumerate(filtered_pairs)}
        connectables = heads.keys() & tails.keys()
        if len(connectables) > 0:
            curr_filtered = []
            for seq in filtered_pairs:
                head_connectable = seq[0] in connectables
                preserve = head_connectable and filtered_pairs[tails[seq[0]]][0] in connectables
                if preserve:
                    curr_filtered.append(seq)
                elif not head_connectable:
                    if seq[-1] in connectables:
                        curr_filtered.append(seq + filtered_pairs[heads[seq[-1]]][1:])
                    else:
                        curr_filtered.append(seq)
            filtered_pairs = curr_filtered
        else:
            break

    return filtered_pairs


def is_simple_reshape(orig_shape, new_shape, mapping: typing.Optional[typing.Dict[int, int]] = None):
    if orig_shape == new_shape:
        if mapping is not None:
            for i in range(len(orig_shape)):
                mapping[i] = i
        return True

    i = 0
    j = 0

    while True:
        if i == len(orig_shape) and j == len(new_shape):
            break
        elif i == len(orig_shape):
            if new_shape[j] == 1:
                j += 1
            else:
                break
        elif j == len(new_shape):
            if orig_shape[i] == 1:
                i += 1
            else:
                break
        elif orig_shape[i] == new_shape[j]:
            if mapping is not None:
                mapping[i] = j
            i += 1
            j += 1
        elif orig_shape[i] == 1:
            i += 1
        elif new_shape[j] == 1:
            j += 1
        else:
            break

    if i != len(orig_shape) or j != len(new_shape):
        return False
    else:
        return True


def reshape_mapping(shape_1, shape_2):
    i = 0
    j = 0
    acc_l = 1
    start_l = 0
    acc_r = 1
    start_r = 0
    mapping_l = []
    mapping_r = []
    sign = None
    while i < len(shape_1) or j < len(shape_2):
        if i < len(shape_1) and j < len(shape_2):
            if start_l == i and start_r == j and shape_1[i] == shape_2[j]:
                mapping_l.append([i])
                mapping_r.append([j])
                acc_l = 1
                acc_r = 1
                i += 1
                j += 1
                start_l = i
                start_r = j
                sign = None
            else:
                if sign in ('l', None):
                    acc_l = shape_1[i] * acc_l
                if sign in ('r', None):
                    acc_r = shape_2[j] * acc_r
                if acc_l == acc_r:
                    mapping_l.append(list(range(start_l, i + 1)))
                    mapping_r.append(list(range(start_r, j + 1)))
                    acc_l = 1
                    acc_r = 1
                    i += 1
                    j += 1
                    start_l = i
                    start_r = j
                    sign = None
                elif acc_l < acc_r:
                    sign = 'l'
                    i += 1
                else:
                    sign = 'r'
                    j += 1
        elif i < len(shape_1):
            assert shape_1[i] == 1
            mapping_l[-1].append(i)
            i += 1
        else:
            assert shape_2[j] == 1
            mapping_r[-1].append(j)
            j += 1
    non_one_mapping_l = []
    non_one_mapping_r = []
    for ml, mr in zip(mapping_l, mapping_r):
        new_ml = [i for i in ml if shape_1[i] != 1]
        new_mr = [j for j in mr if shape_2[j] != 1]
        if len(new_ml) > 0 and len(new_mr) > 0:
            non_one_mapping_l.append(new_ml)
            non_one_mapping_r.append(new_mr)
    return mapping_l, mapping_r, non_one_mapping_l, non_one_mapping_r


def elinimate_sequences(
    graph_converter: CommonGraph,
    filtered_pairs: typing.List[typing.Iterable[ig.Vertex]],
    remove_first_pred: typing.Union[bool, typing.Callable] = False,
    remove_first_node_action: typing.Optional[typing.Callable] = None,
    remove_last_pred: typing.Union[bool, typing.Callable] = True,
    remove_last_node_action: typing.Optional[typing.Callable] = None,
    skip_pred: typing.Union[bool, typing.Callable] = False,
    input_idx: int = 0,
    force_forward_input: bool = False,
):
    remove_ids = []
    actions = []
    for seq in filtered_pairs:
        first_node = seq[0]
        last_node = seq[-1]

        if type(skip_pred) == bool:
            skip = skip_pred
        elif skip_pred is not None:
            skip = skip_pred(seq)

        if skip:
            continue

        if type(remove_first_pred) == bool:
            remove_first = remove_first_pred
            custom_data = None
        elif remove_first_pred is not None:
            remove_first, custom_data = remove_first_pred(seq)

        if type(remove_last_pred) == bool:
            remove_last = remove_last_pred
            custom_data_last = None
        elif remove_last_pred is not None:
            remove_last, custom_data_last = remove_last_pred(seq)

        # If the first node can also be eliminated, then set the previous node as the first node
        if remove_first:
            first_node = graph_converter.graph.vs.find(
                name=graph_converter.tensor_node_map[first_node['op'].inputs[input_idx].name]
            )

        if not remove_last:
            last_node = seq[-2]

        output_idx = 0
        if first_node == seq[0]:
            next_idx = 1
        else:
            next_idx = 0
        output_name = seq[next_idx]['op'].inputs[input_idx].name
        output_idx = first_node['outputs'].index(output_name)

        # We use the forward input tensor under the following circumstances.
        # 1. If the previous node before the sequence is an input node
        # 2. If the first node has multiple outputs and the last node doesn't connect to output nodes
        use_forward_input = False
        if first_node['node_type'] == ExtendedOperator.INPUT_NODE:
            use_forward_input = True

        branch = first_node.outdegree() > 1

        has_output_nodes = False
        for edge in last_node.out_edges():
            target_vertex = edge.target_vertex
            if target_vertex['node_type'] in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE):
                if use_forward_input:
                    # Cannot optimize away ops between i/o nodes
                    skip = True
                else:
                    has_output_nodes = True
                break

        if branch:
            output_outdegree = 0
            for edge in first_node.out_edges():
                target_vertex = edge.target_vertex
                if target_vertex == seq[next_idx]:
                    continue
                if target_vertex['node_type'] in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE):
                    if has_output_nodes and edge['label'] == output_name:
                        output_outdegree += 1
                        break
                else:
                    names = [t.name for t in target_vertex['op'].inputs]
                    if output_name in names:
                        output_outdegree += 1
                        break

            if not has_output_nodes:
                use_forward_input = True
            elif output_outdegree > 0:
                skip = True

        if force_forward_input and not use_forward_input:
            if not has_output_nodes:
                use_forward_input = True
            else:
                skip = True

        if skip:
            continue

        if use_forward_input:
            # Find out the output of the first node in the sequence
            new_output = first_node['outputs'][output_idx]
            assert new_output in graph_converter.tensor_map

            # For each node that is next of the last node, we connect it with the first node
            # Also, the replace the tensors when needed
            graph_converter.replace_next_tensors(last_node, first_node, new_output)
        else:

            # Find out the output of the last node in the sequence
            new_output = last_node['outputs'][0]
            assert new_output in graph_converter.tensor_map

            # For each node that is next of the last node, we connect it with the first node
            graph_converter.connect_next_tensors(last_node, first_node, new_output)

            # Update graph, prepare to drop the output tensor of the intermediate nodes and use the output tensor of
            # the last node instead
            first_node['outputs'][output_idx] = new_output
            if first_node['op'] is not None:
                first_node['op'].outputs[output_idx] = graph_converter.tensor_map[new_output]
            graph_converter.tensor_node_map[new_output] = first_node['name']

        # When the first node is a constant node, we need to set the buffer back
        if first_node['node_type'] == ExtendedOperator.CONSTANT_NODE and not use_forward_input:
            if seq[0]['node_type'] == ExtendedOperator.CONSTANT_NODE:
                old_tensor = graph_converter.tensor_map[seq[0]['name']]
            else:
                old_tensor = seq[0]['op'].inputs[input_idx]
            new_tensor = seq[-1]['op'].outputs[0]
            new_tensor.buffer = old_tensor.buffer

        if remove_first and remove_last:
            # Push the sequence to the removing list
            remove_ids.extend([x.index for x in seq])
        else:
            # Collect actions when removing the first node
            start_index = 0
            end_index = len(seq)
            if not remove_first:
                start_index = 1
                if remove_first_node_action is not None:
                    action = remove_first_node_action(first_node, last_node, custom_data)
                    if action is not None:
                        actions.extend(action)

            if not remove_last:
                end_index = len(seq) - 1
                if remove_last_node_action is not None:
                    action = remove_last_node_action(first_node, last_node, custom_data_last)
                    if action is not None:
                        actions.extend(action)

            # Push the sequence (except the first node) to the removing list
            remove_ids.extend([x.index for x in seq[start_index:end_index]])

    for func, args in actions:
        func(*args)

    graph_converter.graph.delete_vertices(remove_ids)


def expand_op_outputs_in_branches(
    nodes: typing.List[ig.Vertex],
    new_op_func: typing.Callable[[ig.Vertex, ig.Vertex, ig.Vertex], None],
    graph_converter: CommonGraph,
):
    actions = []
    for node in nodes:
        preserve_node = None
        prev_node_name = node['op'].inputs[0].name
        prev_node = graph_converter.graph.vs.find(name=graph_converter.tensor_node_map[prev_node_name])

        # Collect next nodes and choose one to preserve
        next_nodes = []
        for edge in node.out_edges():
            next_node = graph_converter.graph.vs[edge.target]
            if preserve_node is None or next_node['node_type'] == ExtendedOperator.OUTPUT_NODE:
                preserve_node = next_node
            next_nodes.append(next_node)

        # For the filtered nodes, use the cloned op as the previous op
        filtered_nodes = list(set(next_nodes) - set([preserve_node]))
        for next_node in filtered_nodes:
            actions.extend(new_op_func(node, prev_node, next_node))

    # Process actions
    for func, args in actions:
        node = args[0]
        func(*args)


def get_same_padding_args(input_shape, filter_shape, strides, dilation):
    dim = len(input_shape)
    padding = [0] * dim

    for i in range(dim):
        if input_shape[i] % strides[i] == 0:
            padding[i] = max(1 - strides[i] + (filter_shape[i] - 1) * dilation[i], 0)
        else:
            padding[i] = max(1 + (filter_shape[i] - 1) * dilation[i] - (input_shape[i] % strides[i]), 0)

    pad_args = [[0, 0]] + [[x // 2, x - x // 2] for x in padding] + [[0, 0]]

    return pad_args
