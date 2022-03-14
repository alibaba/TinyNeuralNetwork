from ..schemas.tflite import schema_generated as tfl_schema
from . import tflite as tfl
from .base import FUSE_ACTIVATION_MAP, ExtendedOperator
from .graph import CommonGraph


class OPVersioner(object):
    """Sets the version of the OPs in the computation graph"""

    def __init__(self, graph: CommonGraph) -> None:
        """Constructs an OPVersioner object

        Args:
            graph (CommonGraph): The computation graph
        """

        self.graph = graph

    def process(self):
        """The main process function for the whole graph"""
        for node in self.graph.graph.vs:
            if node['node_type'] >= 0:
                self.process_op(node['op'])

    def process_op(self, op: tfl.BaseOperator):
        """Sets the version of the OP

        Args:
            op (tfl.BaseOperator): The operator to be processed
        """

        # Translated from `GetBuiltinOperatorVersion` in
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/versioning/op_version.cc
        if op.op.code == ExtendedOperator.CONV_2D:
            if (
                str(op.inputs[0].dtype) == 'int8'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'int8'
            ):
                op.op.version = 3
            elif (
                str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.DEPTHWISE_CONV_2D:
            if (
                str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 4
            elif (
                str(op.inputs[0].dtype) == 'int8'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'int8'
            ):
                op.op.version = 3
            elif op.dilationHFactor != 1 or op.dilationWFactor != 1:
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.FAKE_QUANT:
            if op.narrowRange:
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.FULLY_CONNECTED:
            if len(op.inputs) == 2:
                op.op.version = 6
            elif op.keepNumDims:
                op.op.version = 5
            elif (
                str(op.inputs[0].dtype) == 'int8'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'int8'
            ):
                op.op.version = 4
            elif (
                str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 3
            elif op.weightsFormat == tfl_schema.FullyConnectedOptionsWeightsFormat.SHUFFLED4x16INT8:
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.GATHER:
            if str(op.inputs[0].dtype) == 'bool':
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.SVDF:
            if str(op.inputs[0].dtype) == 'int8':
                op.op.version = 3
            elif (
                str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[1].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.MUL:
            if (
                op.inputs[0].quantization is not None
                and op.inputs[1].quantization is not None
                and op.outputs[0].quantization is not None
                and op.inputs[0].quantization.scale != 0.0
                and op.inputs[1].quantization.scale != 0.0
                and op.outputs[0].quantization.scale != 0.0
                and op.inputs[0].quantization.scale * op.inputs[1].quantization.scale / op.outputs[0].quantization.scale
                >= 1.0
            ):
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.TRANSPOSE:
            if len(op.inputs[0].shape) > 4:
                op.op.version = 4
            elif str(op.inputs[0].dtype) == 'bool':
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.TRANSPOSE_CONV:
            if str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.LSTM:
            if (
                op.kernelType == tfl_schema.LSTMKernelType.FULL
                and str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[2].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 3
            elif op.kernelType == tfl_schema.LSTMKernelType.BASIC:
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.UNIDIRECTIONAL_SEQUENCE_LSTM:
            if (
                str(op.inputs[0].dtype) == 'float32'
                and str(op.inputs[2].dtype) == 'int8'
                and str(op.outputs[0].dtype) == 'float32'
            ):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.SPLIT:
            if str(op.inputs[1].dtype) == 'int32':
                op.op.version = 3
            elif str(op.inputs[1].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.SPARSE_TO_DENSE:
            if str(op.inputs[2].dtype) in ('int8', 'uint8'):
                op.op.version = 3
            elif str(op.inputs[2].dtype) == 'int64':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.SLICE:
            if str(op.inputs[0].dtype).startswith('<U'):
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.UNPACK:
            if str(op.inputs[0].dtype) == 'int16' or str(op.outputs[0].dtype) == 'int16':
                op.op.version = 4
            elif str(op.inputs[0].dtype) == 'bool':
                op.op.version = 3
            elif str(op.inputs[0].dtype) in ('int8', 'uint8'):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.DEQUANTIZE:
            if str(op.inputs[0].dtype) in ('int16', 'float16'):
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.FLOOR_DIV:
            if str(op.inputs[0].dtype) == 'float32':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.L2_NORMALIZATION:
            if str(op.outputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.RELU:
            if str(op.inputs[0].dtype) in ('int8', 'uint8'):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.STRIDED_SLICE:
            if len(op.inputs[0].shape) > 4:
                op.op.version = 4
            elif str(op.inputs[0].dtype) == 'bool':
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.REVERSE_V2:
            if str(op.outputs[0].dtype) == 'bool':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.RESIZE_BILINEAR:
            if op.halfPixelCenters:
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code in (ExtendedOperator.MINIMUM, ExtendedOperator.MAXIMUM):
            if str(op.inputs[0].dtype) == 'int16' and str(op.outputs[0].dtype) == 'int16':
                op.op.version = 4
            elif (
                len(op.inputs[0].shape) != len(op.inputs[1].shape)
                and max(len(op.inputs[0].shape), len(op.inputs[1].shape)) > 4
            ):
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code == ExtendedOperator.PACK:
            if str(op.inputs[0].dtype) == 'int16' and str(op.outputs[0].dtype) == 'int16':
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.version == ExtendedOperator.TILE:
            if str(op.inputs[0].dtype).startswith('<U'):
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.version in (ExtendedOperator.SPACE_TO_BATCH_ND, ExtendedOperator.BATCH_TO_SPACE_ND):
            if len(op.inputs[0].shape) != 4:
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.version == ExtendedOperator.SUB:
            if (
                len(op.inputs[0].shape) != len(op.inputs[1].shape)
                and max(len(op.inputs[0].shape), len(op.inputs[1].shape)) > 4
            ):
                op.op.version = 3
            elif str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        elif op.op.code in (
            ExtendedOperator.AVERAGE_POOL_2D,
            ExtendedOperator.ADD,
            ExtendedOperator.CONCATENATION,
            ExtendedOperator.MAX_POOL_2D,
            ExtendedOperator.PAD,
            ExtendedOperator.PADV2,
            ExtendedOperator.SOFTMAX,
            ExtendedOperator.SPACE_TO_DEPTH,
            ExtendedOperator.SPLIT_V,
            ExtendedOperator.MEAN,
            ExtendedOperator.SUM,
            ExtendedOperator.REDUCE_MAX,
            ExtendedOperator.REDUCE_MIN,
            ExtendedOperator.RELU6,
            ExtendedOperator.RESIZE_NEAREST_NEIGHBOR,
            ExtendedOperator.TANH,
            ExtendedOperator.LOGISTIC,
            ExtendedOperator.LOG_SOFTMAX,
            ExtendedOperator.TOPK_V2,
            ExtendedOperator.ARG_MAX,
            ExtendedOperator.ARG_MIN,
            ExtendedOperator.EQUAL,
            ExtendedOperator.NOT_EQUAL,
            ExtendedOperator.GREATER,
            ExtendedOperator.GREATER_EQUAL,
            ExtendedOperator.LESS,
            ExtendedOperator.LESS_EQUAL,
            ExtendedOperator.SELECT,
        ):
            if str(op.inputs[0].dtype) == 'int8':
                op.op.version = 2
            else:
                op.op.version = 1
        else:
            op.op.version = 1
