import copy
import functools
import re
import typing
import warnings

import igraph as ig
import numpy as np
from tinynn.util.util import get_logger

from tflite.ActivationFunctionType import ActivationFunctionType

from . import tflite as tfl
from .base import FUSE_ACTIVATION_MAP, ExtendedOperator
from .graph import CommonGraph

log = get_logger(__name__, 'INFO')


class GraphOptimizer(object):
    graph: CommonGraph
    fuse_tensor_count: int
    fuse_attr_count: int

    def __init__(self, graph: CommonGraph) -> None:
        self.graph = graph
        self.fuse_tensor_count = 0
        self.fuse_attr_count = 0

    def create_attr_tensor(self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None):
        if name is None:
            if self.fuse_attr_count == 0:
                name = 'fuse_attr'
            else:
                name = f'fuse_attr_{self.fuse_attr_count}'
            self.fuse_attr_count += 1
        return tfl.Tensor(tensor, name, has_buffer=True, quantization=quantization)

    def create_transform_tensor(self, tensor: tfl.Tensor, name: str = None, quantization: typing.Optional[tfl.QuantizationParameters] = None):
        if name is None:
            if self.fuse_tensor_count == 0:
                name = 'fuse_transform'
            else:
                name = f'fuse_transform_{self.fuse_tensor_count}'
            self.fuse_tensor_count += 1
        return tfl.Tensor(tensor, name, has_buffer=False, quantization=quantization)

    def fuse_conv_fc_bn(self):
        # Find fusable ops
        edges = self.graph.graph.es.select(functools.partial(is_bn_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        actions = []
        for conv, bn, tensor in filtered_pairs:
            # Find out the output of the batch-norm nodes
            new_output = bn['outputs'][0]
            assert new_output in self.graph.tensor_map

            # For each node that is next of a batch-norm node, we connect it with the conv node
            self.graph.connect_next_tensors(bn, conv, new_output)

            # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the batch-norm instead
            conv['outputs'][0] = new_output
            conv['op'].outputs[0] = self.graph.tensor_map[new_output]
            self.graph.tensor_node_map[new_output] = conv['name']
            tensor['name'] = bn['outputs'][0]
            tensor['label'] = bn['outputs'][0]

            bn_activ = bn['op'].fusedActivationFunction
            conv_activ = conv['op'].fusedActivationFunction
            if bn_activ != ActivationFunctionType.NONE and conv_activ == ActivationFunctionType.NONE:
                conv['op'].fusedActivationFunction = bn_activ

            # Collect the arguments of the conv and batch-norm nodes
            weight = conv['op'].inputs[1]
            bias = conv['op'].inputs[2] if len(conv['op'].inputs) > 2 else None
            bn_w, bn_b, bn_mean, bn_var = bn['op'].inputs[1:]
            bn_w, bn_b, bn_mean, bn_var = bn_w.tensor.copy(), bn_b.tensor.copy(), bn_mean.tensor.copy(), bn_var.tensor.copy()
            activ_w = weight.tensor.copy()
            activ_b = bias.tensor.copy() if bias is not None else None
            eps = bn['op'].eps

            # Fuse conv/fc and batch-norm
            new_weight = fuse_bn_weight(eps, bn_w, bn_var, activ_w)
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

            # Update graph, prepare to drop the output tensor of the conv node and use the output tensor of the batch-norm instead
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

    def transform_graph(self):
        # Find transformable ops
        filtered_nodes = self.graph.graph.vs.select(functools.partial(
            is_transformable_node, graph_converter=self.graph.graph))
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

    def fuse_simple_transpose_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_transpose_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            new_perm = fuse_transpose_perms(seq)

            remove_first = np.all(new_perm == np.sort(new_perm))
            return remove_first, new_perm

        def _remove_first_action(first_node, last_node, custom_data):
            # Set fused perm to the first transpose node
            new_perm = custom_data
            new_perm_tensor = self.create_attr_tensor(new_perm)
            action = (self.graph.replace_operator_input, (first_node, 1, new_perm_tensor))
            return [action]

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    def fuse_simple_reshape_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_reshape_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edge
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            first_node, last_node = seq[0], seq[-1]
            new_shape = last_node['op'].inputs[1].tensor
            orig_shape = np.array(first_node['op'].inputs[0].shape, dtype='int32')

            remove_first = np.all(new_shape == orig_shape)
            return remove_first, new_shape

        def _remove_first_action(first_node, last_node, custom_data):
            # Set final shape to the first reshape node
            new_shape = custom_data
            new_shape_tensor = self.create_attr_tensor(np.array(new_shape, dtype='int32'))
            first_node['op'].newShape = new_shape_tensor.tensor
            action = (self.graph.replace_operator_input, (first_node, 1, new_shape_tensor))
            return [action]

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    def fuse_simple_slice_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_slice_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edge
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        def _remove_first_pred(seq):
            fused_info = fuse_slices(seq)

            return False, fused_info

        def _remove_first_action(first_node, last_node, custom_data):
            # Set final shape to the first reshape node
            start, size = custom_data
            start_tensor = self.create_attr_tensor(np.array(start, dtype='int32'))
            size_tensor = self.create_attr_tensor(np.array(size, dtype='int32'))
            actions = [(self.graph.replace_operator_input, (first_node, 1, start_tensor)),
                       (self.graph.replace_operator_input, (first_node, 2, size_tensor))]
            return actions

        elinimate_sequences(self.graph, filtered_pairs, _remove_first_pred, _remove_first_action)

    def cleanup_dead_nodes(self):
        cleanup_nodes = []
        if not self.graph.graph.is_connected('weak'):
            while True:
                for vertex in self.graph.graph.vs:
                    if vertex['node_type'] != ExtendedOperator.OUTPUT_NODE and vertex.outdegree() == 0:
                        if vertex['node_type'] == ExtendedOperator.INPUT_NODE:
                            continue
                        if vertex['node_type'] != ExtendedOperator.CONSTANT_NODE:
                            warnings.warn('Non constant node removed, something must be wrong there')
                            log.warning('-' * 30)
                            log.warning('Info of the deleted node:')
                            log.warning('vertex:', vertex)
                            # edge = self.graph.graph.es.select(name=vertex['outputs'][0])
                            # assert edge is None, f'The edge {vertex["outputs"][0]} exists but the connection to the vertex {vertex["name"]} is broken, \
                            #     probably there have some conflicts in the names of the nodes'
                        cleanup_nodes.append(vertex.index)

                if len(cleanup_nodes) == 0:
                    break

                self.graph.graph.delete_vertices(cleanup_nodes)
                cleanup_nodes.clear()

    def fold_transpose_buffer(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_constant_transpose_fusable_edge, graph_converter=self.graph.graph))
        filtered_pairs = ((self.graph.graph.vs[x.source], self.graph.graph.vs[x.target], x) for x in edges)

        remove_ids = []
        for constant, transpose, tensor in filtered_pairs:
            # Calculate the output of the transposed constant nodes
            constant_tensor = transpose['op'].inputs[0].tensor
            perm_tensor = transpose['op'].inputs[1].tensor
            new_constant = np.transpose(constant_tensor, perm_tensor)
            new_tensor = self.create_attr_tensor(new_constant, quantization=transpose['op'].inputs[0].quantization)
            new_node = self.graph.add_nodes([new_tensor])[0]

            # For each node that is next of a constant transpose node, we connect it with the new constant node
            for out_edge in transpose.out_edges():
                next_node = self.graph.graph.vs[out_edge.target]
                self.graph.graph.add_edge(new_node, next_node, name=new_tensor.name, label=new_tensor.name)
                log.debug(f'NEW EDGE: {new_node["label"]} -> {next_node["label"]} {self.graph.tensor_map[out_edge["name"]]}')
                op = next_node['op']
                for idx in range(len(op.inputs)):
                    if op.inputs[idx].name == transpose['op'].outputs[0].name:
                        op.inputs[idx] = new_tensor

            remove_ids.append(transpose.index)

        # Delete constant transpose nodes
        self.graph.graph.delete_vertices(remove_ids)

    def transpose_to_reshape_pass(self):
        filtered_nodes = self.graph.graph.vs.select(functools.partial(
            is_transformable_transpose_node, graph_converter=self.graph.graph))

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

    def fold_reshape_buffer(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_constant_reshape_fusable_edge, graph_converter=self.graph.graph))
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
                log.debug(f'NEW EDGE: {new_node["label"]} -> {next_node["label"]} {self.graph.tensor_map[out_edge["name"]]}')
                op = next_node['op']
                for idx in range(len(op.inputs)):
                    if op.inputs[idx].name == reshape['op'].outputs[0].name:
                        op.inputs[idx] = new_tensor

            remove_ids.append(reshape.index)

        # Delete constant transpose nodes
        self.graph.graph.delete_vertices(remove_ids)

    def remove_noop_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_ending_with_noop_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        elinimate_sequences(self.graph, filtered_pairs)

    def fuse_wrapped_reshape_within_transpose_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_wrapped_reshape_within_transpose_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

        # Try to fuse the edges
        filtered_pairs = fuse_connected_edges(filtered_pairs)

        # Only TRANSPOSE->RESHAPE->TRANSPOSE is supported here
        filtered_pairs = [seq for seq in filtered_pairs if len(
            seq) == 3 and seq[0]['node_type'] == ExtendedOperator.TRANSPOSE]

        def _skip_pred(seq):
            mid_node = seq[1]
            orig_shape = mid_node['op'].inputs[0].shape
            new_shape = mid_node['op'].outputs[0].shape

            if not is_simple_reshape(orig_shape, new_shape):
                return False

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

    def branch_reshape_expand_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_reshape_branch_edge, graph_converter=self.graph.graph))
        branch_transpose_nodes = list(set(self.graph.graph.vs[edge.source] for edge in edges))

        def _new_reshape(node: ig.Vertex, prev_node: ig.Vertex, next_node: ig.Vertex):
            actions = []

            op = node['op']
            op_out = op.outputs[0]
            op_shape = op.inputs[1]

            prev_idx = prev_node['outputs'].index(op.inputs[0].name)
            prev_op = prev_node['op']
            prev_out = prev_op.outputs[prev_idx]

            new_tensor = self.create_transform_tensor(op_out.tensor.copy(), quantization=op_out.quantization)
            new_shape = self.create_attr_tensor(op_shape.tensor.copy())
            new_op = tfl.ReshapeOperator([prev_out, new_shape], [new_tensor], new_shape.tensor)
            self.graph.add_operator(new_op)

            next_indices = []
            for i, t in enumerate(next_node['op'].inputs):
                if t.name == op_out.name:
                    actions.append((self.graph.replace_operator_input, (next_node, i, new_tensor)))
                    next_indices.append(i)

            assert len(next_indices) > 0, f'{op_out.name} not in {[t.name for t in next_node["op"].inputs]}'

            return actions

        expand_op_outputs_in_branches(branch_transpose_nodes, _new_reshape, self.graph)

    def branch_transpose_expand_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_transpose_branch_edge, graph_converter=self.graph.graph))
        branch_transpose_nodes = list(set(self.graph.graph.vs[edge.source] for edge in edges))

        def _new_transpose(node: ig.Vertex, prev_node: ig.Vertex, next_node: ig.Vertex):
            actions = []

            op = node['op']
            op_out = op.outputs[0]
            op_perm = op.inputs[1]

            prev_idx = prev_node['outputs'].index(op.inputs[0].name)
            prev_op = prev_node['op']
            prev_out = prev_op.outputs[prev_idx]

            new_tensor = self.create_transform_tensor(op_out.tensor.copy(), quantization=op_out.quantization)
            new_perm = self.create_attr_tensor(op_perm.tensor.copy())
            new_op = tfl.TransposeOperator([prev_out, new_perm], [new_tensor])
            self.graph.add_operator(new_op)

            next_indices = []
            for i, t in enumerate(next_node['op'].inputs):
                if t.name == op_out.name:
                    actions.append((self.graph.replace_operator_input, (next_node, i, new_tensor)))
                    next_indices.append(i)

            assert len(next_indices) > 0, f'{op_out.name} not in {[t.name for t in next_node["op"].inputs]}'

            return actions

        expand_op_outputs_in_branches(branch_transpose_nodes, _new_transpose, self.graph)

    def elementwise_op_transpose_passthrough_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_transpose_elementwise_op_edge, graph_converter=self.graph.graph))
        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.TRANSPOSE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        for node in unique_nodes:
            op = node['op']
            if node['node_type'] == ExtendedOperator.CONCATENATION:
                input_indices = range(len(op.inputs))
            elif node['node_type'] == ExtendedOperator.SPLIT:
                input_indices = (1, )
            elif node['node_type'] in (ExtendedOperator.ADD,
                                       ExtendedOperator.SUB,
                                       ExtendedOperator.MUL,
                                       ExtendedOperator.DIV):
                input_indices = range(2)
            else:
                input_indices = range(1)

            prev_nodes = []
            cand_perms = dict()
            prev_output_indices = []
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)
                prev_output_indices.append(prev_node['outputs'].index(prev_node_name))

                if prev_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    perm = tuple(prev_node['op'].inputs[1].tensor.tolist())
                    cand_perms.setdefault(perm, 0)
                    cand_perms[perm] += 1

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

                if next_node['node_type'] == ExtendedOperator.TRANSPOSE:
                    perm = tuple(np.argsort(next_node['op'].inputs[1].tensor).tolist())
                    cand_perms.setdefault(perm, 0)
                    cand_perms[perm] += 1

            cur_transpose_size = sum(cand_perms.values())
            new_transpose_size = len(prev_nodes) + len(next_nodes) - cur_transpose_size

            # Skip if the number of transpose nodes is not decreasing
            if len(next_nodes) == 0 or new_transpose_size >= cur_transpose_size:
                continue

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for node in out_nodes:
                del self.graph.tensor_map[node['outputs'][0]]
                del self.graph.tensor_node_map[node['outputs'][0]]

            perm = max(cand_perms.items(), key=lambda x: x[1])[0]
            perm_arr = np.array(perm, dtype='int32')
            inv_perm_arr = np.argsort(perm_arr).astype('int32')

            for prev_node, prev_idx, next_idx in zip(prev_nodes, input_indices, prev_output_indices):
                prev_out = prev_node['op'].outputs[next_idx]
                perm_tensor = self.create_attr_tensor(inv_perm_arr)
                prev_new_out = self.create_transform_tensor(np.transpose(
                    prev_out.tensor, inv_perm_arr), quantization=prev_out.quantization)
                self.graph.add_operator(tfl.TransposeOperator([prev_out, perm_tensor], [prev_new_out]))
                actions.append((self.graph.replace_operator_input, (node, prev_idx, prev_new_out, True)))

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                perm_tensor = self.create_attr_tensor(perm_arr)
                new_out = self.create_transform_tensor(np.transpose(
                    op_out.tensor, inv_perm_arr), quantization=op_out.quantization)

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                self.graph.add_operator(tfl.TransposeOperator([new_out, perm_tensor], [op_out]))

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

            # OP specific dim handling logic
            if node['node_type'] == ExtendedOperator.CONCATENATION:
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

    def elementwise_op_reshape_passthrough_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_reshape_elementwise_op_edge, graph_converter=self.graph.graph))
        pairs = ((self.graph.graph.vs[edge.source], self.graph.graph.vs[edge.target]) for edge in edges)
        filtered_nodes = (k[0] if k[0]['node_type'] != ExtendedOperator.RESHAPE else k[1] for k in pairs)
        unique_nodes = list(set(filtered_nodes))

        actions = []
        remove_edges = []
        remove_vertices = []
        for node in unique_nodes:
            op = node['op']
            dim_indice = None
            if node['node_type'] == ExtendedOperator.CONCATENATION:
                input_indices = range(len(op.inputs))
                dim_indice = op.axis
            elif node['node_type'] == ExtendedOperator.SPLIT:
                input_indices = (1, )
                dim_indice = op.inputs[0].tensor[0]
            elif node['node_type'] == ExtendedOperator.SPLIT_V:
                input_indices = range(1)
                dim_indice = op.inputs[2].tensor[0]
            elif node['node_type'] in (ExtendedOperator.ADD,
                                       ExtendedOperator.SUB,
                                       ExtendedOperator.MUL,
                                       ExtendedOperator.DIV):
                input_indices = range(2)
            else:
                input_indices = range(1)

            prev_nodes = []
            cand_shapes = dict()
            cand_next_shapes = dict()
            for i in input_indices:
                prev_node_name = op.inputs[i].name
                prev_node = self.graph.graph.vs.find(name=self.graph.tensor_node_map[prev_node_name])
                prev_nodes.append(prev_node)

                if prev_node['node_type'] == ExtendedOperator.RESHAPE:
                    mapping = dict()
                    if is_simple_reshape(prev_node['op'].inputs[0].shape, prev_node['op'].outputs[0].shape, mapping):
                        continue

                    new_dim = None
                    if dim_indice is not None:
                        rev_mapping = {v: k for k, v in mapping.items()}
                        if dim_indice not in rev_mapping:
                            continue
                        new_dim = rev_mapping[dim_indice]

                    shape = tuple(prev_node['op'].inputs[0].shape)
                    shape = tuple(x if i != new_dim else -1 for i, x in enumerate(shape))
                    cand_shapes.setdefault(shape, 0)
                    cand_shapes[shape] += 1

                    next_shape = tuple(prev_node['op'].outputs[0].shape)
                    next_shape = tuple(x if i != dim_indice else -1 for i, x in enumerate(next_shape))
                    cand_next_shapes.setdefault(next_shape, 0)
                    cand_next_shapes[next_shape] += 1

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

                if next_node['node_type'] == ExtendedOperator.RESHAPE:
                    mapping = dict()
                    if is_simple_reshape(next_node['op'].inputs[0].shape, next_node['op'].outputs[0].shape, mapping):
                        continue

                    new_dim = None
                    if dim_indice is not None:
                        if dim_indice not in mapping:
                            continue
                        new_dim = mapping[dim_indice]

                    shape = tuple(next_node['op'].outputs[0].shape)
                    shape = tuple(x if i != new_dim else -1 for i, x in enumerate(shape))
                    cand_shapes.setdefault(shape, 0)
                    cand_shapes[shape] += 1

                    next_shape = tuple(next_node['op'].inputs[0].shape)
                    next_shape = tuple(x if i != dim_indice else -1 for i, x in enumerate(next_shape))
                    cand_next_shapes.setdefault(next_shape, 0)
                    cand_next_shapes[next_shape] += 1

            cur_reshape_size = max(cand_shapes.values())
            cur_next_reshape_size = max(cand_next_shapes.values())
            full_size = len(prev_nodes) + len(next_nodes)

            # Skip if not wrapped by reshapes
            if len(next_nodes) == 0 or cur_reshape_size < full_size or cur_next_reshape_size < full_size:
                continue

            remove_edges.extend([x.index for x in next_edges])
            remove_vertices.extend([x.index for x in out_nodes])

            for node in out_nodes:
                del self.graph.tensor_map[node['outputs'][0]]
                del self.graph.tensor_node_map[node['outputs'][0]]

            prev_shape = max(cand_shapes.items(), key=lambda x: x[1])[0]
            next_shape = max(cand_next_shapes.items(), key=lambda x: x[1])[0]

            for i, prev_node in enumerate(prev_nodes):
                prev_out = prev_node['op'].outputs[0]
                prev_new_out = self.create_transform_tensor(np.reshape(
                    prev_out.tensor, prev_shape), quantization=prev_out.quantization)
                shape_tensor = self.create_attr_tensor(np.array(prev_new_out.shape, dtype='int32'))
                self.graph.add_operator(tfl.ReshapeOperator([prev_out, shape_tensor], [
                                        prev_new_out], newShape=shape_tensor.tensor))
                actions.append((self.graph.replace_operator_input, (node, i, prev_new_out)))

            tensor_node_dict = {}
            for i, op_out in enumerate(op.outputs):
                new_out = self.create_transform_tensor(np.reshape(
                    op_out.tensor, next_shape), quantization=op_out.quantization)
                shape_tensor = self.create_attr_tensor(np.array(new_out.shape, dtype='int32'))

                # Update relations
                if op_out.name in self.graph.tensor_node_map:
                    del self.graph.tensor_node_map[op_out.name]
                self.graph.tensor_node_map[new_out.name] = node['name']
                self.graph.tensor_map[new_out.name] = new_out
                node['outputs'][i] = new_out.name
                op.outputs[i] = new_out

                self.graph.add_operator(tfl.ReshapeOperator([new_out, shape_tensor], [op_out], shape_tensor.tensor))

                tensor_node_dict[op_out.name] = self.graph.graph.vs.find(name=self.graph.tensor_node_map[op_out.name])

            # OP specific dim handling logic
            if node['node_type'] in ExtendedOperator.CONCATENATION:
                new_axis = prev_shape.index(-1)
                op.axis = new_axis
            elif node['node_type'] in ExtendedOperator.SPLIT_V:
                new_dim = prev_shape.index(-1)
                new_dim_tensor = self.create_attr_tensor(new_dim)
                actions.append(self.graph.replace_operator_input, (node, 2, new_dim_tensor))
            elif node['node_type'] in ExtendedOperator.SPLIT:
                new_dim = prev_shape.index(-1)
                new_dim_tensor = self.create_attr_tensor(new_dim)
                actions.append(self.graph.replace_operator_input, (node, 0, new_dim_tensor))

            for edge in next_edges:
                source = tensor_node_dict[edge['name']]
                self.graph.graph.add_edge(source, edge.target_vertex, name=edge['name'], label=edge['name'])

        self.graph.graph.delete_vertices(remove_vertices)
        self.graph.graph.delete_edges(remove_edges)

        # Process actions
        for func, args in actions:
            node = args[0]
            func(*args)

    def fuse_bmm_add_pass(self):
        edges = self.graph.graph.es.select(functools.partial(
            is_bmm_add_edge, graph_converter=self.graph.graph))
        filtered_pairs = [[self.graph.graph.vs[x.source], self.graph.graph.vs[x.target]] for x in edges]

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
        sorted_ops = [(nodes[0]['op'], nodes[1]['op'])
                      for nodes in sorted(ops, key=lambda x: int(re.search(r'\d+', x[1]['name'])[0]))]

        # Delete nodes before transformation in the graph
        self.graph.graph.delete_vertices(remove_ids)

        for (bmm, add), mapping in zip(sorted_ops, restore_mapping):
            input_tensor = bmm.inputs[0]
            weight_tensor = bmm.inputs[1]
            bias_tensor = add.inputs[1]
            output_tensor = add.outputs[0]

            ops = []

            input_as_2d = self.create_transform_tensor(input_tensor.tensor[0])
            input_2d_shape = self.create_attr_tensor(np.array(input_as_2d.shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([input_tensor, input_2d_shape], [input_as_2d], input_2d_shape.tensor))

            weight_t = self.create_transform_tensor(np.transpose(weight_tensor.tensor))
            weight_perm = self.create_attr_tensor(np.array([1, 0], dtype='int32'))
            ops.append(tfl.TransposeOperator([weight_tensor, weight_perm], [weight_t]))

            output_as_2d = self.create_transform_tensor(output_tensor.tensor[0])
            ops.append(tfl.FullyConnectedOperator([input_as_2d, weight_t, bias_tensor], [
                       output_as_2d], fusedActivationFunction=add.fusedActivationFunction))

            output_3d_shape = self.create_attr_tensor(np.array(output_tensor.shape, dtype='int32'))
            ops.append(tfl.ReshapeOperator([output_as_2d, output_3d_shape], [output_tensor], output_3d_shape.tensor))

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
                transposed = self.create_transform_tensor(np.transpose(
                    last_tensor.tensor, nhwc2nchw_perm), quantization=last_tensor.quantization)
                transpose_op = tfl.TransposeOperator([last_tensor, nhwc2nchw_perm_tensor], [transposed])
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

    def optimize(self):
        # Input/output passes
        self.output_list_unpack_pass()
        self.input_transpose_pass()

        # Transpose, Reshape and NO-OP cleanup
        self.branch_reshape_expand_pass()
        self.fuse_simple_reshape_pass()
        self.branch_transpose_expand_pass()
        self.fuse_simple_transpose_pass()
        self.remove_noop_pass()
        self.fuse_wrapped_reshape_within_transpose_pass()

        # Buffer folding, which is needed by the fusion passes below
        for _ in range(2):
            self.fold_reshape_buffer()
            self.fold_transpose_buffer()

        # OP fusion passes before transformation
        self.fuse_conv_fc_bn()
        self.fuse_activation()

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

        # Branch transpose cleanup
        for _ in range(3):
            self.elementwise_op_transpose_passthrough_pass()
            self.branch_transpose_expand_pass()
            self.fuse_simple_transpose_pass()

        # Other cleanups
        self.fuse_simple_slice_pass()
        self.remove_noop_pass()
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

        # Final cleanup
        self.cleanup_dead_nodes()


def is_bn_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] in (ExtendedOperator.GENERIC_CONV, ExtendedOperator.GENERIC_DECONV, ExtendedOperator.FULLY_CONNECTED) \
        and target_vertex['node_type'] == ExtendedOperator.BATCH_NORM and source_vertex.outdegree() == 1 \
        and target_vertex['op'].inputs[1].buffer is not None and target_vertex['op'].inputs[2].buffer is not None \
        and source_vertex['op'].inputs[1].buffer is not None \
        and (target_vertex['op'].fusedActivationFunction == ActivationFunctionType.NONE or
             source_vertex['op'].fusedActivationFunction in (ActivationFunctionType.NONE, target_vertex['op'].fusedActivationFunction))


def is_activ_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] in (ExtendedOperator.FULLY_CONNECTED, ExtendedOperator.GENERIC_CONV,
                                          ExtendedOperator.ADD, ExtendedOperator.SUB, ExtendedOperator.MUL,
                                          ExtendedOperator.DIV, ExtendedOperator.MAX_POOL_2D, ExtendedOperator.AVERAGE_POOL_2D) \
        and target_vertex['node_type'] in (ExtendedOperator.RELU, ExtendedOperator.RELU6) \
        and source_vertex['op'].fusedActivationFunction == ActivationFunctionType.NONE \
        and source_vertex.outdegree() == 1


def is_transformable_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return vertex['node_type'] <= ExtendedOperator.BATCH_NORM and vertex.outdegree() >= 1


def is_transformable_transpose_node(vertex: ig.Vertex, graph_converter: ig.Graph):
    return vertex['node_type'] == ExtendedOperator.TRANSPOSE and vertex.outdegree() >= 1 \
        and is_transpose_same_to_reshape_op(vertex['op'])


def is_transpose_elementwise_op_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return ((source_vertex['node_type'] == ExtendedOperator.TRANSPOSE and
             (is_elementwise_unary_op(target_vertex['node_type'], target_vertex['op']) or
              is_elementwise_binary_op(target_vertex['node_type'], target_vertex['op']))) or
            (target_vertex['node_type'] == ExtendedOperator.TRANSPOSE and
                (is_elementwise_unary_op(source_vertex['node_type'], source_vertex['op']) or
                 is_elementwise_binary_op(source_vertex['node_type'], source_vertex['op'])))) \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_reshape_elementwise_op_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return ((source_vertex['node_type'] == ExtendedOperator.RESHAPE
             and (is_elementwise_unary_op(target_vertex['node_type'], target_vertex['op'])
                  or is_elementwise_binary_op(target_vertex['node_type'], target_vertex['op'])))
            or (target_vertex['node_type'] == ExtendedOperator.RESHAPE
                and (is_elementwise_unary_op(source_vertex['node_type'], source_vertex['op'])
                     or is_elementwise_binary_op(source_vertex['node_type'], source_vertex['op'])))) \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_elementwise_unary_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return op_code in (ExtendedOperator.RELU,
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
                       ExtendedOperator.SOFTMAX,
                       ExtendedOperator.LOG_SOFTMAX,
                       ExtendedOperator.HARD_SWISH,
                       ExtendedOperator.LEAKY_RELU)


def is_elementwise_binary_op(op_code: ExtendedOperator, op: tfl.BaseOperator):
    return (op_code in (ExtendedOperator.CONCATENATION,
                        ExtendedOperator.ADD,
                        ExtendedOperator.SUB,
                        ExtendedOperator.MUL,
                        ExtendedOperator.DIV) and
            len(op.inputs) >= 2 and
            op.inputs[0].tensor.ndim == op.inputs[1].tensor.ndim) \
        or (op_code in (ExtendedOperator.SPLIT,
                        ExtendedOperator.SPLIT_V))


def is_ending_with_noop_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex.outdegree() == 1 and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name \
        and ((target_vertex['node_type'] == ExtendedOperator.RESHAPE and
              target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape) or
             (target_vertex['node_type'] == ExtendedOperator.TRANSPOSE and
                 (np.diff(target_vertex['op'].inputs[1].tensor) == 1).all()) or
             (target_vertex['node_type'] == ExtendedOperator.PAD and
                 target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape) or
             (target_vertex['node_type'] == ExtendedOperator.SLICE and
                 target_vertex['op'].inputs[0].shape == target_vertex['op'].outputs[0].shape) or
             (target_vertex['node_type'] == ExtendedOperator.CAST and
                 target_vertex['op'].inDataType == target_vertex['op'].outDataType))


def is_bmm_add_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.BATCH_MATMUL \
        and target_vertex['node_type'] == ExtendedOperator.ADD \
        and source_vertex['op'].inputs[0].tensor.ndim == 3 \
        and source_vertex['op'].inputs[0].shape[0] == 1 \
        and source_vertex['op'].inputs[1].tensor.ndim == 2 \
        and target_vertex['op'].inputs[1].tensor.ndim == 1 \
        and target_vertex['op'].inputs[1].shape[0] == source_vertex['op'].inputs[1].shape[-1] \
        and source_vertex.outdegree() == 1 and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_wrapped_reshape_within_transpose_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return ((target_vertex['node_type'] == ExtendedOperator.TRANSPOSE and
             source_vertex['node_type'] == ExtendedOperator.RESHAPE) or
            (source_vertex['node_type'] == ExtendedOperator.TRANSPOSE and
                target_vertex['node_type'] == ExtendedOperator.RESHAPE)) \
        and source_vertex.outdegree() == 1 and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_slice_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.SLICE and source_vertex.outdegree() == 1 \
        and target_vertex['node_type'] == ExtendedOperator.SLICE and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_transpose_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.TRANSPOSE and source_vertex.outdegree() == 1 \
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_reshape_branch_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.RESHAPE and source_vertex.outdegree() > 1 \
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_transpose_branch_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.TRANSPOSE and source_vertex.outdegree() > 1 \
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_reshape_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.RESHAPE and source_vertex.outdegree() == 1 \
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_constant_transpose_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.CONSTANT_NODE and source_vertex.outdegree() == 1 \
        and target_vertex['node_type'] == ExtendedOperator.TRANSPOSE and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_constant_reshape_fusable_edge(edge: ig.Edge, graph_converter: ig.Graph):
    source_vertex = graph_converter.vs[edge.source]
    target_vertex = graph_converter.vs[edge.target]
    return source_vertex['node_type'] == ExtendedOperator.CONSTANT_NODE and source_vertex.outdegree() == 1 \
        and target_vertex['node_type'] == ExtendedOperator.RESHAPE and target_vertex.outdegree() >= 1 \
        and source_vertex['outputs'][0] == target_vertex['op'].inputs[0].name


def is_transpose_same_to_reshape_op(op: tfl.BaseOperator):
    num_elements = np.prod(op.inputs[0].shape)

    input_shape = np.array(op.inputs[0].shape, dtype='int32')
    output_shape = np.array(op.outputs[0].shape, dtype='int32')

    if np.all(input_shape[input_shape != 1] == output_shape[output_shape != 1]):
        input_tensor = np.arange(num_elements).reshape(input_shape)
        perm = op.inputs[1].tensor
        new_tensor = np.transpose(input_tensor, perm)

        return np.all(new_tensor.flatten() == input_tensor.flatten())
    else:
        return False


def fuse_bn_weight(eps, scale, var, weight):
    while weight.ndim > scale.ndim:
        scale = scale[:, None]
    while weight.ndim > var.ndim:
        var = var[:, None]

    eps = np.array(eps, dtype='float32')

    return weight * scale / np.sqrt(var + eps, dtype='float32')


def fuse_bn_bias(eps, scale, var, mean, bn_b, activ_b):
    if scale.ndim > 1:
        scale = scale.flatten()
    if var.ndim > 1:
        var = var.flatten()

    eps = np.array(eps, dtype='float32')

    if activ_b is not None:
        if activ_b.shape != mean.shape and activ_b.ndim == 1 and activ_b.size == 1:
            activ_b = activ_b.repeat(mean.size)
        return ((activ_b - mean) * scale) / (np.sqrt(var + eps, dtype='float32')) + bn_b
    else:
        return ((- mean) * scale) / (np.sqrt(var + eps, dtype='float32')) + bn_b


def fuse_slices(seq: typing.Iterable[ig.Vertex]):
    cur_start = None
    cur_size = None
    for node in seq:
        assert node['node_type'] == ExtendedOperator.SLICE
        next_start = node['op'].inputs[1].tensor
        next_size = node['op'].inputs[2].tensor
        if cur_start is None and cur_size is None:
            cur_start = next_start
            cur_size = next_size
        else:
            cur_start += next_start
            cur_size = np.min((cur_size, next_size), axis=0)
    return cur_start, cur_size


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
                for old, item in zip(old_shape, cur_perm):
                    if old == new_shape_padded[new_idx]:
                        next_perm.append(item)
                        new_idx += 1
                cur_perm = np.argsort(next_perm)

    return cur_perm


def fuse_connected_edges(filtered_pairs: typing.List[typing.Iterable[ig.Vertex]]) -> typing.List[typing.Iterable[ig.Vertex]]:
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
            if mapping:
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


def elinimate_sequences(graph_converter: CommonGraph, filtered_pairs: typing.List[typing.Iterable[ig.Vertex]],
                        remove_first_pred: typing.Union[bool, typing.Callable] = False,
                        remove_first_node_action: typing.Optional[typing.Callable] = None,
                        remove_last_pred: typing.Union[bool, typing.Callable] = True,
                        remove_last_node_action: typing.Optional[typing.Callable] = None,
                        skip_pred: typing.Union[bool, typing.Callable] = False):
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
                name=graph_converter.tensor_node_map[first_node['op'].inputs[0].name])

        if not remove_last:
            last_node = seq[-2]

        # We use the forward input tensor under the following circumstances.
        # 1. If the previous node before the sequence is an input node
        # 2. If the first node has multiple outputs
        use_forward_input = False
        if first_node['node_type'] == ExtendedOperator.INPUT_NODE or first_node.outdegree() > 1:
            use_forward_input = True

        if use_forward_input:
            # Find out the output of the first node in the sequence
            new_output = first_node['outputs'][0]
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

            # Update graph, prepare to drop the output tensor of the intermediate nodes and use the output tensor of the last node instead
            first_node['outputs'][0] = new_output
            if first_node['op'] is not None:
                first_node['op'].outputs[0] = graph_converter.tensor_map[new_output]
            graph_converter.tensor_node_map[new_output] = first_node['name']

        if remove_first and remove_last:
            # When the first node is a constant node, we need to set the buffer back
            if first_node['node_type'] == ExtendedOperator.CONSTANT_NODE and not use_forward_input:
                old_tensor = seq[0]['op'].inputs[0]
                new_tensor = seq[-1]['op'].outputs[0]
                new_tensor.buffer = old_tensor.buffer

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


def expand_op_outputs_in_branches(nodes: typing.List[ig.Vertex], new_op_func: typing.Callable[[ig.Vertex, ig.Vertex, ig.Vertex], None],
                                  graph_converter: CommonGraph):
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
