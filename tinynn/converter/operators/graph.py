import os
import typing
import warnings

import flatbuffers
import igraph as ig

from . import tflite as tfl
from .base import ExtendedOperator

from tinynn.util.util import get_logger

log = get_logger(__name__, 'INFO')


class CommonGraph(object):
    graph: ig.Graph
    tensor_map: typing.Dict[str, tfl.Tensor]
    tensor_node_map: typing.Dict[str, str]
    iterable_map: typing.Dict[str, typing.List[str]]
    inputs: typing.List[str]
    outputs: typing.List[str]
    input_transpose: typing.List[bool]
    output_transpose: typing.Union[typing.List[typing.Optional[bool]], typing.Optional[bool]]
    node_op_counter: int

    def __init__(self) -> None:
        self.graph = ig.Graph(directed=True)
        self.tensor_map = dict()
        self.tensor_node_map = dict()
        self.iterable_map = dict()
        self.inputs = []
        self.outputs = []
        self.input_transpose = []
        self.output_transpose = None
        self.node_op_counter = 0
        self.q_mapping = {}

    def add_iterable_pair(
        self, input_names: typing.List[str], output_names: typing.List[str], key: typing.Optional[str] = None
    ):
        """Adds the tensor mapping for a ListConstruct tensor

        Args:
            input_names (typing.List[str]): The names of the input tensors
            output_names (typing.List[str]): The names of the output tensors
            key (typing.Literal['input', 'output'], optional): Which side is used as key. Defaults to None.
        """

        if key == 'input' or len(input_names) == 1 and len(output_names) > 1:
            list_name = input_names[0]
            self.iterable_map.setdefault(list_name, [])
            self.iterable_map[list_name].extend(output_names)
        elif key == 'output' or len(input_names) > 1 and len(output_names) == 1:
            list_name = output_names[0]
            self.iterable_map.setdefault(list_name, [])
            self.iterable_map[list_name].extend(input_names)
        else:
            assert False, "You should specify key == 'input' or 'output'"

    def has_nested_names(self, key: str) -> bool:
        """Whether a tensor has nested tensor (names)

        Args:
            key (str): The name of the tensor

        Returns:
            bool: Whether it is a ListConstruct tensor
        """

        return key in self.iterable_map

    def get_list_expanded_names(self, key: str) -> typing.List[str]:
        """Get the names of the nested tensors of a ListConstruct tensor

        Args:
            key (str): The name of the ListConstruct tensor

        Returns:
            typing.List[str]: The names of the nested tensors
        """

        return self.iterable_map[key]

    def check_tensor(self, name: str, node_type: ExtendedOperator, tensor: tfl.Tensor) -> ig.Vertex:
        """Checks whether the node with the tensor as the output already exists

        Args:
            name (str): The name of the tensor
            node_type (ExtendedOperator): The type of the node
            tensor (tfl.Tensor): The tensor

        Returns:
            ig.Vertex: The node that produces the tensor
        """
        node_name = self.tensor_node_map[name]
        node = self.graph.vs.find(name=node_name)
        assert name in self.tensor_map, f"tensor {name} is in nodes map, but not in tensors map"
        # assert node["node_type"] == node_type, f"tensor {name} already exists, but with a different type"
        assert id(self.tensor_map[name]) == id(tensor), f"tensor {name} already exists"
        return node

    def add_nodes(
        self, tensors: typing.List[tfl.Tensor], node_type=ExtendedOperator.CONSTANT_NODE
    ) -> typing.List[ig.Vertex]:
        """Add a list of nodes (usually special ones) with the tensors

        Args:
            tensors (typing.List[tfl.Tensor]): The output tensors of the nodes
            node_type ([type], optional): The type of the node. Defaults to ExtendedOperator.CONSTANT_NODE.

        Returns:
            ig.Vertex: The newly-created nodes
        """

        nodes = []
        for t in tensors:
            if node_type in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE):
                tensor_name = t.name + '_output'
                if tensor_name in self.tensor_map:
                    i = 1
                    while True:
                        tensor_name = f'{t.name}_output_{i}'
                        if tensor_name in self.tensor_map:
                            i += 1
                        else:
                            break
            else:
                tensor_name = t.name

            if tensor_name in self.tensor_node_map:
                nodes.append(self.check_tensor(tensor_name, node_type, t))
            else:
                node = self.graph.add_vertex(
                    node_type=node_type,
                    outputs=[tensor_name],
                    label=ExtendedOperator(node_type).type_name(),
                    name=tensor_name,
                )
                self.tensor_map[tensor_name] = t
                self.tensor_node_map[tensor_name] = node['name']
                nodes.append(node)
        return nodes

    def add_node(self, tensors: typing.List[tfl.Tensor], tfl_op: tfl.BaseOperator, output_exists: bool = False):
        """Add a node (usually a op node) with the output tensors

        Args:
            tensors (typing.List[tfl.Tensor]): The output tensors of the node
            tfl_op (tfl.BaseOperator): The op to be added
            output_exists (bool, optional): Whether the output may already exists. Defaults to False.

        Returns:
            [type]: [description]
        """

        output_names = [t.name for t in tfl_op.outputs]
        node_unique_name = f'__tinynn_op_{self.node_op_counter}__'
        self.node_op_counter += 1
        if tfl_op.op.custom_code is not None:
            node = self.graph.add_vertex(
                node_type=tfl_op.op.code,
                custom_type=tfl_op.op.custom_code,
                outputs=output_names,
                op=tfl_op,
                label=tfl_op.type_name(),
                name=node_unique_name,
            )
        else:
            node = self.graph.add_vertex(
                node_type=tfl_op.op.code,
                outputs=output_names,
                op=tfl_op,
                label=tfl_op.type_name(),
                name=node_unique_name,
            )

        log.debug(f'NEW VERTEX:  {node["op"].type_name()}[{node["name"]}] {node["op"].inputs} -> {node["op"].outputs}')

        for t in tensors:
            if not output_exists:
                assert (
                    t.name not in self.tensor_node_map
                ), f"output tensor ({t.name}) should not be in the nodes map at this time"
                self.tensor_map[t.name] = t
            else:
                if t.name in self.tensor_map:
                    assert (
                        self.tensor_map[t.name] == t
                    ), f"output tensor ({t.name}) has changed during graph reconstruction"
                else:
                    log.debug(f'tensor node map add {t.name} during transformation')
                    self.tensor_map[t.name] = t

            self.tensor_node_map[t.name] = node['name']
        return node

    def add_outputs(self, names: typing.List[str], node_type=ExtendedOperator.OUTPUT_NODE):
        """Add the output nodes with the names given

        Args:
            names (typing.List[str]): The names of the output nodes to be created
        """

        if len(names) > 0:
            output_tensors = list(map(lambda x: self.tensor_map[x], names))
            output_nodes = self.add_nodes(output_tensors, node_type)
            for idx, (name, output_node) in enumerate(zip(names, output_nodes)):
                current_node = self.graph.vs.find(name=self.tensor_node_map[name])
                edge = self.graph.add_edge(current_node, output_node, name=output_node["outputs"][0], label=name)
                log.debug(
                    f'NEW EDGE: {current_node["label"]} -> {output_node["label"]} {self.tensor_map[edge["name"]]}'
                )

    def add_operator(self, tfl_op: tfl.BaseOperator, transform: bool = False):
        """Add a new operator to the graph

        Args:
            tfl_op (tfl.BaseOperator): The operator be added
            transform (bool, optional): Whether it is created by a transformable node. Defaults to False.
        """
        input_nodes = self.add_nodes(tfl_op.inputs)
        current_node = self.add_node(tfl_op.outputs, tfl_op, transform)
        for idx, input_node in enumerate(input_nodes):
            edge = self.graph.add_edge(
                input_node, current_node, name=tfl_op.inputs[idx].name, label=tfl_op.inputs[idx].name
            )
            log.debug(f'NEW EDGE: {input_node["label"]} -> {current_node["label"]} {self.tensor_map[edge["name"]]}')

        output_names = set(self.outputs).intersection(set([t.name for t in tfl_op.outputs]))
        self.add_outputs(output_names)

    def try_restore_edges(self, mapping: typing.List[typing.Tuple[str, str]]):
        """Try to restore the edges between nodes

        Args:
            mapping (typing.List[typing.Tuple[str, str]]): A list of mapping (edge name, target node nam)
        """

        for edge_name, node_name in mapping:
            cand = self.graph.vs.select(name=node_name)
            # Only restore when the node exists
            if cand:
                next_node = cand[0]
                prev_node = self.graph.vs.find(name=self.tensor_node_map[edge_name])
                edge = self.graph.add_edge(prev_node, next_node, name=edge_name, label=edge_name)
                log.debug(f'NEW EDGE: {prev_node["label"]} -> {next_node["label"]} {self.tensor_map[edge["name"]]}')

    def remove_operator_input(
        self, node: ig.Vertex, input_idx: int, return_ids: bool = False, skip: int = 0
    ) -> typing.Optional[typing.List[int]]:
        """Remove an input tensor in a op node

        Args:
            node (ig.Vertex): An op node
            input_idx (int): the index of the input tensor
            return_ids (bool): Return the ids instead of removing the edges. Defaults to False.
            skip (int): Number of items to skip

        Returns:
            typing.Optional[typing.List[int]]: The edges to be removed if return_ids is True, otherwise None
        """

        old_tensor = node['op'].inputs[input_idx]
        assert old_tensor.name in self.tensor_map

        remove_edges = []
        for edge in node.in_edges():
            start = self.graph.vs[edge.source]
            for i in range(len(start['outputs'])):
                if start['outputs'][i] == old_tensor.name and edge['name'] == old_tensor.name:
                    if skip > 0:
                        skip -= 1
                        continue
                    remove_edges.append(edge.index)
                    break
            if len(remove_edges) > 0:
                break

        if return_ids:
            return remove_edges
        else:
            self.graph.delete_edges(remove_edges)

    def replace_operator_input(
        self, node: ig.Vertex, input_idx: int, new_tensor: tfl.Tensor, return_ids: bool = False, skip: int = 0
    ) -> typing.Optional[typing.List[int]]:
        """Use a new input tensor in a op node

        Args:
            node (ig.Vertex): An op node
            input_idx (int): the index of the input tensor
            new_tensor (tfl.Tensor): The tensor to be be used
            return_ids (bool): Return the ids instead of removing the edges. Defaults to False.
            skip (int): Number of items to skip

        Returns:
            typing.Optional[typing.List[int]]: The edges to be removed if return_ids is True, otherwise None
        """

        remove_edges = self.remove_operator_input(node, input_idx, return_ids=True, skip=skip)

        node['op'].inputs[input_idx] = new_tensor
        new_node = self.add_nodes([new_tensor])[0]
        edge = self.graph.add_edge(new_node, node, name=new_tensor.name, label=new_tensor.name)
        log.debug(f'NEW EDGE: {new_node["label"]} -> {node["label"]} {self.tensor_map[edge["name"]]}')

        if return_ids:
            return remove_edges
        else:
            self.graph.delete_edges(remove_edges)

    def append_operator_input(self, node: ig.Vertex, new_tensor: tfl.Tensor):
        """Add a new input tensor to a op node

        Args:
            node (ig.Vertex): An op node
            new_tensor (tfl.Tensor): The tensor to be added
        """
        node['op'].inputs.append(new_tensor)
        new_node = self.add_nodes([new_tensor])[0]
        edge = self.graph.add_edge(new_node, node, name=new_tensor.name, label=new_tensor.name)
        log.debug(f'NEW EDGE: {new_node["label"]} -> {node["label"]} {self.tensor_map[edge["name"]]}')

    def remove_operator(self, tfl_op: tfl.BaseOperator):
        tensor_edge = self.graph.es.find(name=tfl_op.outputs[0].name)
        op_node = tensor_edge.source
        self.graph.delete_vertices([op_node.index])

    def remove_operators(self, tfl_ops: typing.List['tfl.BaseOperator']):
        indices = []
        for tfl_op in tfl_ops:
            tensor_edge = self.graph.es.find(name=tfl_op.outputs[0].name)
            op_node = tensor_edge.source
            indices.append(op_node.index)
        self.graph.delete_vertices(indices)

    def connect_next_tensors(
        self,
        find_node: ig.Vertex,
        connect_node: ig.Vertex,
        tensor_name: str,
        skips_nodes: typing.Optional[typing.List[str]] = None,
    ):
        """Add edges between `connect_node` and the next nodes of `find_node` with the name `tensor_name`

        Args:
            find_node ([ig.Vertex]): The node to search for next nodes
            connect_node ([ig.Vertex]): The node to connect the next nodes with
            tensor_name ([str]): The name of the edge (tensor)
            skip_nodes ([typing.Optional[typing.List[str]]]): The name of the next nodes to skip
        """
        for next_tensor in find_node.out_edges():
            next_op = self.graph.vs[next_tensor.target]
            if skips_nodes is not None and next_op['name'] in skips_nodes:
                continue
            if next_op['node_type'] not in (ExtendedOperator.OUTPUT_NODE, ExtendedOperator.UNUSED_NODE):
                assert (
                    tensor_name == next_tensor['name']
                ), f'next tensor name mismatches: {tensor_name} vs {next_tensor["name"]}'
                self.graph.add_edge(connect_node, next_op, name=tensor_name, label=tensor_name)
            else:
                assert next_tensor['name'].startswith(
                    tensor_name + '_output'
                ), f'output tensor and node name mismatches: {tensor_name} vs {next_tensor["name"]}'
                self.graph.add_edge(connect_node, next_op, name=next_tensor['name'], label=tensor_name)

            log.debug(f'NEW EDGE: {connect_node["label"]} -> {next_op["label"]} {self.tensor_map[next_tensor["name"]]}')

    def replace_next_tensors(
        self,
        find_node: ig.Vertex,
        connect_node: ig.Vertex,
        tensor_name: str,
        skips_nodes: typing.Optional[typing.List[str]] = None,
    ):
        """A variant of connect_next_tensors that also replace the tensors in the next nodes

        Args:
            find_node ([ig.Vertex]): The node to search for next nodes
            connect_node ([ig.Vertex]): The node to connect the next nodes with
            tensor_name ([str]): The name of the edge (tensor)
            skip_nodes ([typing.Optional[typing.List[str]]]): The name of the next nodes to skip
        """
        orig_name = find_node['outputs'][0]

        for next_tensor in find_node.out_edges():
            next_op = self.graph.vs[next_tensor.target]
            if skips_nodes is not None and next_op['name'] in skips_nodes:
                continue
            if next_op['node_type'] != ExtendedOperator.OUTPUT_NODE:
                assert (
                    orig_name == next_tensor['name']
                ), f'next tensor name mismatches: {tensor_name} vs {next_tensor["name"]}'
                op = next_op['op']
                for idx, t in enumerate(op.inputs):
                    if t.name == orig_name:
                        op.inputs[idx] = self.tensor_map[tensor_name]
                self.graph.add_edge(connect_node, next_op, name=tensor_name, label=tensor_name)
            else:
                assert False, 'replace_next_tensors where last_node.next is an output node is not supported'

            log.debug(f'NEW EDGE: {connect_node["label"]} -> {next_op["label"]} {self.tensor_map[next_tensor["name"]]}')
            log.debug(f'{next_op["label"]} {next_op["op"].inputs} {next_op["op"].outputs}')

    def visualize(self, hide_constants=True):
        """Plot the TinyNeuralNetwork graph

        Args:
            hide_constants (bool, optional): Hide constants in the plot. Defaults to True.
        """

        self.check()
        import matplotlib.pyplot as plt

        _, axs = plt.subplots()

        if hide_constants:
            nodes = self.graph.vs.select(node_type_ne=ExtendedOperator.CONSTANT_NODE)
            subgraph = self.graph.induced_subgraph(nodes)
        else:
            subgraph = self.graph

        visual_style = {}
        visual_style["vertex_label_size"] = 5
        visual_style["vertex_label"] = subgraph.vs["outputs"]
        visual_style["layout"] = "drl"
        visual_style["bbox"] = (800, 800)
        visual_style["margin"] = 20
        ig.plot(subgraph, target=axs, **visual_style)
        axs.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.show()

    def check(self):
        """Checks whether the graph is in a good state"""

        assert self.graph.is_dag(), "The graph is not a DAG"
        assert self.graph.is_directed(), "The graph is not directed"

        # For simple NNs, the following checks should also pass
        # Unfortunately, it is hard to tell whether the NN is simple or not.
        # assert self.graph.is_simple(), "The graph has multiple edges between at least one pair of nodes"
        # assert self.graph.is_connected('weak'), "The graph is not connected"

    def collect_operators(self) -> typing.List[tfl.BaseOperator]:
        """Collect ops

        Returns:
            typing.List[tfl.BaseOperator]: operators with the numbered index
        """

        ids = self.graph.topological_sorting()
        nodes = (self.graph.vs[idx] for idx in ids)
        filtered_nodes = (node for node in nodes if node['node_type'] >= 0)
        ops: typing.List[tfl.BaseOperator] = (x['op'] for x in filtered_nodes)

        log.debug('Collecting operators...')
        result = []
        for idx, op in enumerate(ops):
            log.debug(f'[{idx}] {op.type_name()} {op.inputs} -> {op.outputs}')
            op.op.index = idx
            op.tfl_inputs_idx = [x.index for x in op.inputs]
            op.tfl_outputs_idx = [x.index for x in op.outputs]
            result.append(op)
        return result

    def collect_tensor_buffers(
        self,
    ) -> typing.Tuple[typing.List[tfl.Tensor], typing.List[tfl.Buffer], typing.List[int], typing.List[int]]:
        """ Collect tensors, buffers and I/O indices

        Returns:
            typing.Tuple[typing.List[tfl.Tensor], typing.List[tfl.Buffer], typing.List[int], typing.List[int]]: \
                tensors, buffers with the numbered index and I/O indices
        """

        labels = set(self.graph.es['label'])

        tensor_idx = 0
        buffer_idx = 1

        tensors = []
        buffers = [tfl.Buffer(bytes(0))]
        input_idx = [-1] * len(self.inputs)
        output_idx = [-1] * len(self.outputs)
        for label in labels:
            tensor: tfl.Tensor = self.tensor_map[label]
            if tensor.index != -1:
                if tensor.is_variable:
                    tensor.buffer.index = 0
                tensor.index = tensor_idx
                tensor_idx += 1

                tensors.append(tensor)

                if tensor.buffer is not None and tensor.is_variable is False:
                    tensor.buffer.index = buffer_idx
                    buffer_idx += 1

                    buffers.append(tensor.buffer)

            if label in self.inputs:
                item_indices = [i for i, x in enumerate(self.inputs) if x == label]
                for item_idx in item_indices:
                    input_idx[item_idx] = tensor.index

            if label in self.outputs:
                item_indices = [i for i, x in enumerate(self.outputs) if x == label]
                for item_idx in item_indices:
                    output_idx[item_idx] = tensor.index

        missing_inputs = [name for name, _ in filter(lambda x: x[1] < 0, zip(self.inputs, input_idx))]
        missing_outputs = [name for name, _ in filter(lambda x: x[1] < 0, zip(self.outputs, output_idx))]

        assert len(missing_outputs) == 0, f'Some output nodes are missing: {missing_outputs}'

        if len(missing_inputs) != 0:
            warnings.warn(f'Some input nodes are missing: {missing_inputs}, will try to add them into graph')
            for name in missing_inputs:
                tensor = self.tensor_map[name]
                tensor.index = tensor_idx
                tensor_idx += 1
                tensors.append(tensor)
                item_idx = self.inputs.index(name)
                input_idx[item_idx] = tensor.index

        return tensors, buffers, input_idx, output_idx

    def convert(self, tflite_path: str):
        """Convert from the TinyNeuralNetwork Graph to the tflite model

        Args:
            tflite_path ([str]): Path of the generated tflite model
        """

        # Collect multiple data to build a tflite model
        tensors, buffers, input_idx, output_idx = self.collect_tensor_buffers()
        ops = self.collect_operators()

        # Start flatbuffer
        builder = flatbuffers.Builder(0)

        # Write data into flatbuffer
        tensor_offsets = [t.build(builder) for t in tensors]
        op_offsets = [op.build(builder) for op in ops]
        opcode_offsets = [op.op.build(builder) for op in ops]
        buffer_offsets = [buffer.build(builder) for buffer in buffers]

        # Build Subgraph
        subgraph = tfl.SubGraph()
        subgraph.tensors.extend(tensor_offsets)
        subgraph.inputs.extend(input_idx)
        subgraph.outputs.extend(output_idx)
        subgraph.operators.extend(op_offsets)

        # Build Model
        model = tfl.Model()
        model.buffers.extend(buffer_offsets)
        model.subgraphs.append(subgraph.build(builder))
        model.opcodes.extend(opcode_offsets)
        model = model.build(builder)
        builder.Finish(model, b"TFL3")

        # Finish Model
        tflite_model = builder.Output()

        # Check output directory
        tflite_dir = os.path.abspath(os.path.dirname(tflite_path))
        os.makedirs(tflite_dir, exist_ok=True)

        # Write to file
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
