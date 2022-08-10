import copy
import typing

import flatbuffers
import numpy as np
import torch

from ...schemas.tflite import schema_generated as tflite

Offset = int


class OpCode(object):
    code: int
    version: int
    index: int
    tfl_opcode: Offset

    def __init__(self, code: int, version: int, custom_code: typing.Optional[str] = None):
        self.code = code
        self.version = version
        self.index = 0
        self.custom_code = custom_code
        self.tfl_opcode = 0

    def build(self, builder: flatbuffers.Builder) -> Offset:
        custom_code = None
        if self.custom_code is not None:
            custom_code = create_string(builder, tflite.OperatorCode.CustomCode, self.custom_code)

        tflite.OperatorCodeStart(builder)
        if self.code < tflite.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
            tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, self.code)
        else:
            tflite.OperatorCodeAddBuiltinCode(builder, self.code)
        tflite.OperatorCodeAddVersion(builder, self.version)

        if custom_code is not None:
            tflite.OperatorCodeAddCustomCode(builder, custom_code)

        self.tfl_opcode = tflite.OperatorCodeEnd(builder)

        return self.tfl_opcode


class BaseOperator(object):
    inputs: typing.List['Tensor']
    outputs: typing.List['Tensor']
    op: OpCode
    tfl_op: Offset
    tfl_inputs_idx: typing.Iterable[int]
    tfl_outputs_idx: typing.Iterable[int]
    extra_hints: typing.Dict[str, typing.Any]

    def __init__(self, op: int, inputs: typing.List['Tensor'], outputs: typing.List['Tensor'], op_version: int = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.op = OpCode(op, op_version)

        self.tfl_op = 0
        self.tfl_inputs_idx = []
        self.tfl_outputs_idx = []

        self.extra_hints = {}

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op

    def type_name(self) -> str:
        return type(self).__name__.replace('Operator', '')


class QuantizationParameters:
    scale: typing.Union[float, typing.List[float]]
    zero_point: typing.Union[int, typing.List[int]]
    tfl_quant_args: Offset

    def __init__(
        self, scale: typing.Union[float, typing.List[float]], zero_point: int, dim: typing.Optional[int] = None
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.dim = dim

        self.tfl_quant_args = 0

    def build(self, builder: flatbuffers.Builder) -> Offset:
        if isinstance(self.scale, float):
            scale = create_numpy_array(builder, tflite.QuantizationParameters.Scale, [self.scale], 'float32')
        else:
            scale = create_numpy_array(builder, tflite.QuantizationParameters.Scale, self.scale, 'float32')

        if isinstance(self.zero_point, int):
            zero_point = create_numpy_array(
                builder, tflite.QuantizationParameters.ZeroPoint, [self.zero_point], 'int64'
            )
        else:
            zero_point = create_numpy_array(builder, tflite.QuantizationParameters.ZeroPoint, self.zero_point, 'int64')

        tflite.QuantizationParametersStart(builder)
        tflite.QuantizationParametersAddMin(builder, 0)
        tflite.QuantizationParametersAddMax(builder, 0)
        tflite.QuantizationParametersAddScale(builder, scale)
        tflite.QuantizationParametersAddZeroPoint(builder, zero_point)

        if self.dim is not None:
            tflite.QuantizationParametersAddQuantizedDimension(builder, self.dim)

        self.tfl_quant_args = tflite.QuantizationParametersEnd(builder)

        return self.tfl_quant_args

    def __repr__(self) -> str:
        return f'scale={self.scale}, zero_point={self.zero_point}'


class Buffer(object):
    data: typing.Union[bytearray, bytes]
    index: int
    tfl_buffer: Offset

    def __init__(self, data: typing.Union[bytearray, bytes]):
        self.data = data
        self.index = 0

        self.tfl_buffer = 0

    def build(self, builder: flatbuffers.Builder) -> Offset:
        if len(self.data) != 0:
            data = create_byte_array(builder, tflite.Buffer.Data, self.data)
        else:
            data = 0
        tflite.BufferStart(builder)
        tflite.BufferAddData(builder, data)
        self.tfl_buffer = tflite.BufferEnd(builder)

        return self.tfl_buffer


class FakeQuantTensor(object):
    def __init__(self, tensor, scale, zero_point, dim=None) -> None:
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.dim = dim


class Tensor(object):
    tensor: np.ndarray
    name: str
    quantization: typing.Optional[QuantizationParameters]
    buffer: typing.Optional[Buffer]
    dtype: np.dtype
    shape: typing.Iterable[int]
    tfl_tensor: int

    def __init__(
        self,
        tensor: typing.Iterable,
        name: str,
        quantization: QuantizationParameters = None,
        has_buffer: bool = True,
        dtype: str = None,
        is_variable: bool = False,
        asymmetric: bool = True,
        q_type: type = np.uint8,
    ):
        self.quantization = None
        self.name = name
        self.index = 0
        self.is_variable = is_variable

        if type(tensor) == FakeQuantTensor:
            self.quantization = QuantizationParameters(tensor.scale, tensor.zero_point, tensor.dim)
            tensor = tensor.tensor

        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.data

        if type(tensor).__module__ == 'numpy':
            self.tensor = tensor
        elif type(tensor) == torch.Tensor:
            assert tensor.is_contiguous, "Tensor should be contiguous"
            if tensor.dtype == torch.quint8:
                self.tensor = torch.int_repr(tensor.detach()).numpy()
                if q_type == np.uint8:
                    self.quantization = QuantizationParameters(tensor.q_scale(), tensor.q_zero_point())
                else:
                    if not asymmetric:
                        sym_u8_offset = 128
                        assert tensor.q_zero_point() == sym_u8_offset, (
                            "As for symmetric quantization, the zero point of the u8 tensors should be"
                            f" {sym_u8_offset}, but got {tensor.q_zero_point()}. This could happen if you didn't train"
                            " the model after QAT preparation, or the OP is not supported in symmetric quantization"
                            " (e.g. sigmoid)"
                        )
                    else:
                        sym_u8_offset = tensor.q_zero_point()
                    scale = tensor.q_scale()
                    self.tensor = (self.tensor.astype(np.int32) - 128).astype(np.int8)
                    if q_type == np.int16:
                        scale = scale * 255 / 65535
                        self.tensor = np.round(self.tensor.astype(np.float32) / 255 * 65535).astype(np.int16)
                    self.quantization = QuantizationParameters(scale, sym_u8_offset - 128)
            elif tensor.dtype == torch.qint8:
                self.tensor = torch.int_repr(tensor.detach()).numpy()
                if q_type == np.uint8:
                    if asymmetric:
                        asym_s8_offset = 0
                        assert tensor.q_zero_point() == asym_s8_offset, (
                            "As for asymmetric quantization, the zero point of the s8 tensors should be"
                            f" {asym_s8_offset}, but got {tensor.q_zero_point()}. "
                        )
                    else:
                        asym_s8_offset = tensor.q_zero_point()
                    self.tensor = self.tensor.view(np.uint8) + 128
                    self.quantization = QuantizationParameters(tensor.q_scale(), asym_s8_offset + 128)
                else:
                    if tensor.qscheme() in (torch.per_tensor_symmetric, torch.per_tensor_affine):
                        self.quantization = QuantizationParameters(tensor.q_scale(), tensor.q_zero_point())
                    else:
                        assert tensor.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine)

                        scales = tensor.q_per_channel_scales().tolist()
                        zero_points = tensor.q_per_channel_zero_points().tolist()
                        dim = tensor.q_per_channel_axis()

                        if dim < 0:
                            dim += tensor.dim()

                        assert all((t == 0 for t in zero_points)), (
                            'As for per-channel quantization, "                             "the zero point of the s8'
                            f' tensors should be 0, but got ${zero_points}'
                        )

                        self.quantization = QuantizationParameters(scales, zero_points, dim)
            else:
                self.tensor = tensor.detach().numpy()
        elif type(tensor) == torch.Size:
            self.tensor = np.asarray(tensor, dtype='int32')
        elif type(tensor) in (tuple, list):
            self.tensor = np.asarray(tensor, dtype=dtype)
        else:
            assert False, f"unrecognized tensor type {type(tensor).__name__}"

        if has_buffer:
            self.buffer = Buffer(self.tensor.tobytes())
        else:
            self.buffer = None

        self.dtype = self.tensor.dtype
        self.shape = self.tensor.shape

        if quantization is not None:
            self.quantization = copy.deepcopy(quantization)

        self.tfl_tensor = 0

    def __repr__(self) -> str:
        return f'{self.name}: {self.dtype}{self.shape}'

    def reinterpret_as(self, new_type: typing.Union[type, np.dtype]):
        self.tensor = self.tensor.view(new_type)
        self.dtype = self.tensor.dtype

    def build(self, builder: flatbuffers.Builder) -> Offset:
        name = create_string(builder, tflite.Tensor.Name, self.name)
        shape = create_numpy_array(builder, tflite.Tensor.Shape, self.shape)
        dtype = numpy_tflite_dtype_mappings[str(self.dtype)]

        buffer = 0
        if self.buffer is not None:
            buffer = self.buffer.index

        quantization = 0
        if self.quantization is not None:
            quantization = self.quantization.build(builder)

        tflite.TensorStart(builder)
        tflite.TensorAddBuffer(builder, buffer)
        tflite.TensorAddIsVariable(builder, self.is_variable)
        tflite.TensorAddName(builder, name)
        tflite.TensorAddShape(builder, shape)
        tflite.TensorAddType(builder, dtype)
        tflite.TensorAddQuantization(builder, quantization)
        self.tfl_tensor = tflite.TensorEnd(builder)

        return self.tfl_tensor


class OptionalTensor(Tensor):
    def __init__(self):
        self.index = -1
        self.quantization = None
        self.name = '__tinynn_optional_tensor__'
        self.is_variable = False
        self.tensor = None
        self.shape = None
        self.dtype = None
        self.buffer = None

    def __repr__(self) -> str:
        return 'OptionalTensor'

    def build(self, builder: flatbuffers.Builder) -> Offset:
        raise Exception('Could not build an optional tensor')


class SubGraph(object):
    tensors: typing.List[Offset]
    inputs: typing.List[int]
    outputs: typing.List[int]
    operators: typing.List[Offset]
    tfl_subgraph: int

    def __init__(self):
        self.tensors = []
        self.inputs = []
        self.outputs = []
        self.operators = []

        self.tfl_subgraph = 0

    def build(self, builder: flatbuffers.Builder) -> Offset:
        inputs = create_numpy_array(builder, tflite.SubGraph.Inputs, self.inputs)
        outputs = create_numpy_array(builder, tflite.SubGraph.Outputs, self.outputs)
        operators = create_offset_vector(builder, tflite.SubGraph.Operators, self.operators)
        tensors = create_offset_vector(builder, tflite.SubGraph.Tensors, self.tensors)
        name = create_string(builder, tflite.SubGraph.Name, "main_graph")

        tflite.SubGraphStart(builder)
        tflite.SubGraphAddInputs(builder, inputs)
        tflite.SubGraphAddOutputs(builder, outputs)
        tflite.SubGraphAddName(builder, name)
        tflite.SubGraphAddTensors(builder, tensors)
        tflite.SubGraphAddOperators(builder, operators)
        self.tfl_subgraph = tflite.SubGraphEnd(builder)

        return self.tfl_subgraph


class Model(object):
    buffers: typing.List[Offset]
    opcodes: typing.List[Offset]
    subgraphs: typing.List[Offset]
    tfl_model: Offset

    def __init__(self):
        self.buffers = []
        self.opcodes = []
        self.subgraphs = []

        self.tfl_model = 0

    def build(self, builder: flatbuffers.Builder) -> Offset:
        buffers = create_offset_vector(builder, tflite.Model.Buffers, self.buffers)
        opcodes = create_offset_vector(builder, tflite.Model.OperatorCodes, self.opcodes)
        subgraphs = create_offset_vector(builder, tflite.Model.Subgraphs, self.subgraphs)
        description = create_string(builder, tflite.Model.Description, "TinyNeuralNetwork Converted.")
        version = 3

        tflite.ModelStart(builder)
        tflite.ModelAddBuffers(builder, buffers)
        tflite.ModelAddDescription(builder, description)
        tflite.ModelAddVersion(builder, version)
        tflite.ModelAddOperatorCodes(builder, opcodes)
        tflite.ModelAddSubgraphs(builder, subgraphs)
        self.tfl_model = tflite.ModelEnd(builder)

        return self.tfl_model


def create_offset_vector(builder: flatbuffers.Builder, prop: typing.Callable, vec: typing.Iterable):
    if type(vec) not in (tuple, list):
        assert False, "type of vec unexpected, expected: list or tuple"
    elif type(vec) == tuple:
        vec = list(vec)

    prop_name = prop.__name__
    cls_name = prop.__qualname__.split('.')[0]
    func_name = f'{cls_name}Start{prop_name}Vector'
    if not hasattr(tflite, func_name):
        assert False, f"invalid prop is given, {prop.__qualname__}"

    start_vec_func = getattr(tflite, func_name)
    start_vec_func(builder, len(vec))
    for item in reversed(vec):
        builder.PrependUOffsetTRelative(item)

    try:
        end = builder.EndVector(len(vec))
    except TypeError:
        end = builder.EndVector()
    return end


def create_numpy_array(builder: flatbuffers.Builder, prop: typing.Callable, vec: typing.Iterable, dtype: str = 'int32'):
    if type(vec) not in (tuple, list, torch.Size) and type(vec).__module__ != 'numpy':
        assert False, "type of vec unexpected, expected: list or tuple or ndarray"

    prop_name = prop.__name__
    cls_name = prop.__qualname__.split('.')[0]
    func_name = f'{cls_name}Start{prop_name}Vector'
    if not hasattr(tflite, func_name):
        assert False, f"invalid prop is given, {prop.__qualname__}"

    arr = np.asarray(vec, dtype=dtype)
    return builder.CreateNumpyVector(arr)


def create_string(builder: flatbuffers.Builder, prop: typing.Callable, val: str):
    if type(val) != str:
        assert False, "type of val unexpected, expected: str"

    prop_name = prop.__name__
    cls_name = prop.__qualname__.split('.')[0]
    func_name = f'{cls_name}Add{prop_name}'
    if not hasattr(tflite, func_name):
        assert False, f"invalid prop is given, {prop.__qualname__}"

    return builder.CreateString(val)


def create_byte_array(builder: flatbuffers.Builder, prop: typing.Callable, val: typing.Union[bytes, bytearray]):
    if type(val) not in (bytearray, bytes):
        assert False, "type of val unexpected, expected: bytes or bytearray"

    prop_name = prop.__name__
    cls_name = prop.__qualname__.split('.')[0]
    func_name = f'{cls_name}Start{prop_name}Vector'
    if not hasattr(tflite, func_name):
        assert False, f"invalid prop is given, {prop.__qualname__}"

    return builder.CreateByteVector(val)


numpy_tflite_dtype_mappings = {
    'bool': tflite.TensorType.BOOL,
    'int16': tflite.TensorType.INT16,
    'int32': tflite.TensorType.INT32,
    'int64': tflite.TensorType.INT64,
    'int8': tflite.TensorType.INT8,
    'uint8': tflite.TensorType.UINT8,
    'float16': tflite.TensorType.FLOAT16,
    'float32': tflite.TensorType.FLOAT32,
    'float64': tflite.TensorType.FLOAT64,
}

torch_tflite_dtype_mappings = {
    torch.bool: tflite.TensorType.BOOL,
    torch.int16: tflite.TensorType.INT16,
    torch.int32: tflite.TensorType.INT32,
    torch.int64: tflite.TensorType.INT64,
    torch.qint8: tflite.TensorType.INT8,
    torch.quint8: tflite.TensorType.UINT8,
    torch.float16: tflite.TensorType.FLOAT16,
    torch.float32: tflite.TensorType.FLOAT32,
    torch.float64: tflite.TensorType.FLOAT64,
}

OptionalTensorInstance = OptionalTensor()
