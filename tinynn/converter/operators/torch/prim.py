import torch

import numpy as np

from . import PrimOperatorConverter
from .. import tflite as tfl

from tinynn.util.util import get_logger

log = get_logger(__name__, 'INFO')


class PrimConstantConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        if attrs is not None:
            v, vk = attrs.get('value', (None, None))
            vt = v.dtype if hasattr(v, "dtype") else type(v).__name__
            log.debug(f'{node.kind()} {self.input_names} -> {self.output_names} {vk} {vt}')
            self.output_tensors.append(v)
        else:
            self.output_tensors = None


class PrimTupleConstructConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        self.output_tensors.append(tuple(self.input_tensors))


class PrimListConstructConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        self.output_tensors.append(list(self.input_tensors))
        graph_converter.add_iterable_pair(self.input_names, self.output_names, 'output')


class PrimListUnpackConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        assert type(self.input_tensors[0]) in (list, tuple)
        assert len(self.input_tensors[0]) == len(self.output_names)
        self.output_tensors.extend(self.input_tensors[0])

        try:
            name = self.input_names[0]
            input_names = graph_converter.get_list_expanded_names(name)
            inputs = self.to_tfl_tensors(input_names, self.input_tensors[0], graph_converter=graph_converter)
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            # Connect the tensors with a no-op that will be removed in the optimize passes
            for i, o in zip(inputs, outputs):
                s = np.array(o.shape, dtype='int32')
                st = self.create_attr_tensor(s)
                graph_converter.add_operator(tfl.ReshapeOperator([i, st], [o], o.shape))

        except KeyError:
            # The input is not tracked, nothing needs to be done to the graph converter
            pass


class PrimGetAttrConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        name, name_type = attrs.get('name', (None, None))
        if name is not None and name_type == 's':
            v = getattr(self.input_tensors[0], name)
            self.output_tensors.append(v)
        else:
            assert False, f"prim::GetAttr({self.output_names[0]}) needs attribute `name` with type str"


class PrimNumToTensorConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        assert type(self.input_tensors[0]) in (int, float)
        assert len(self.input_tensors) == len(self.output_names)
        t = torch.tensor(self.input_tensors[0])
        if t.dtype == torch.int64:
            log.warning(
                f'{self.output_names[0]} is of type int64, which is unsupported in TFLite, trying to downcast to int32'
            )
            t = t.to(dtype=torch.int32)
        if t.dtype == torch.float64:
            log.warning(
                f'{self.output_names[0]} is of type float64, which is unsupported in TFLite, trying to downcast to'
                ' float32'
            )
            t = t.to(dtype=torch.float32)
        self.output_tensors.append(t)


class PrimIfConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        assert len(self.input_tensors) == 1
        assert isinstance(self.input_tensors[0], (bool, int))
        assert len(self.output_names) == 0

        cond_var_name = self.input_names[0]
        assert cond_var_name not in graph_converter.tensor_map, 'Dynamic control flow is not supported'

        blocks = list(node.blocks())
        assert len(blocks) == 2

        if self.input_tensors[0] in (True, 1):
            self.output_nodes.extend(blocks[0].nodes())
        else:
            self.output_nodes.extend(blocks[1].nodes())


class PrimGetItemConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        input_tensor = self.input_tensors[0]
        idx = self.input_tensors[1]

        self.output_tensors.append(input_tensor[idx])


class PrimLenConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        input_tensor = self.input_tensors[0]

        self.output_tensors.append(len(input_tensor))


class PrimConstantChunkConverter(PrimOperatorConverter):
    def parse(self, node, attrs, args, graph_converter):
        chunks, chunks_type = attrs.get('chunks', (None, None))
        dim, dim_type = attrs.get('dim', (None, None))

        if chunks is None or chunks_type != 'i':
            assert False, f"prim::ConstantChunk({self.output_names[0]}) needs attribute `chunks` with type int"

        if dim is None or dim_type != 'i':
            assert False, f"prim::ConstantChunk({self.output_names[0]}) needs attribute `dim` with type int"

        v = torch.chunk(self.input_tensors[0], chunks, dim)
        self.output_tensors.extend(v)

        # Graph operations only take place when the input tensor is tracked
        if self.input_names[0] in graph_converter.tensor_map:
            outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

            if dim < 0:
                dim += len(self.input_tensors[0].shape)

            dim_size = self.input_tensors[0].size(dim)
            if chunks > dim_size:
                chunks = dim_size

            input_tensor = self.find_or_create_input(0, graph_converter)
            dim_tensor = self.create_attr_tensor(np.array([dim], dtype='int32'))

            if dim_size % chunks != 0:
                size_splits = np.array([t.size(dim) for t in self.output_tensors], dtype='int32')
                chunks = len(size_splits)
                split_tensor = self.create_attr_tensor(size_splits)
                graph_converter.add_operator(
                    tfl.SplitVOperator([input_tensor, split_tensor, dim_tensor], outputs, chunks)
                )
            else:
                graph_converter.add_operator(tfl.SplitOperator([dim_tensor, input_tensor], outputs, chunks))
