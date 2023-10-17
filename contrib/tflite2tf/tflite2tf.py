from tinynn.converter.utils.tflite import parse_model
from tinynn.converter.schemas.tflite import schema_generated as tflite

import os
import typing


## Utility functions
def input_name(op: tflite.Operator, idx: int, transform_tensors: typing.Set[int]):
    tensor_id = op.Inputs(idx)
    if tensor_id in transform_tensors:
        return f'tensor_{tensor_id}'
    else:
        return f'self.tensor_{tensor_id}'


def output_name(op: tflite.Operator, idx: int, transform_tensors: typing.Set[int]):
    tensor_id = op.Outputs(idx)
    assert tensor_id in transform_tensors
    return f'tensor_{tensor_id}'


def handle_fused_act(act, data_line):
    if act == tflite.ActivationFunctionType.RELU:
        data_line = f'tf.nn.relu({data_line})'
    elif act == tflite.ActivationFunctionType.RELU6:
        data_line = f'tf.nn.relu6({data_line})'
    elif act != tflite.ActivationFunctionType.NONE:
        print('Fused act not supported:', act)
        exit(1)
    return data_line


## OP parsing functions
def parse_pad(op: tflite.Operator, transform_tensors: typing.Set[int]):
    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.pad({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)})'
    )
    return line


def parse_add(op: tflite.Operator, transform_tensors: typing.Set[int]):
    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.math.add({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)})'
    )
    return line


def parse_reshape(op: tflite.Operator, transform_tensors: typing.Set[int]):
    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.reshape({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)})'
    )
    return line


def parse_resize(op: tflite.Operator, transform_tensors: typing.Set[int]):
    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.image.resize({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)})'
    )
    return line


def parse_relu(op: tflite.Operator, transform_tensors: typing.Set[int]):
    line = f'{output_name(op, 0, transform_tensors)} = tf.nn.relu({input_name(op, 0, transform_tensors)})'
    return line


def parse_depth2space(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.DepthToSpaceOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.DepthToSpaceOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    upsample_scale = opt.BlockSize()

    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.nn.depth_to_space({input_name(op, 0, transform_tensors)},'
        f' {upsample_scale})'
    )
    return line


def parse_conv2d(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.Conv2DOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.Conv2DOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    strides = [opt.StrideH(), opt.StrideW()]
    padding = "SAME" if opt.Padding() == tflite.Padding.SAME else "VALID"
    dilations = [opt.DilationHFactor(), opt.DilationWFactor()]

    data_line = (
        f'tf.nn.bias_add(tf.nn.conv2d({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)}, strides={strides}, padding="{padding}", dilations={dilations}),'
        f' {input_name(op, 2, transform_tensors)})'
    )
    data_line = handle_fused_act(opt.FusedActivationFunction(), data_line)
    line = f'{output_name(op, 0, transform_tensors)} = {data_line}'

    return line


def parse_depthwiseconv2d(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.DepthwiseConv2DOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.DepthwiseConv2DOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    strides = [1, opt.StrideH(), opt.StrideW(), 1]
    padding = "SAME" if opt.Padding() == tflite.Padding.SAME else "VALID"
    dilations = [opt.DilationHFactor(), opt.DilationWFactor()]

    data_line = (
        f'tf.nn.bias_add(tf.nn.depthwise_conv2d({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)}, strides={strides}, padding="{padding}", dilations={dilations}),'
        f' {input_name(op, 2, transform_tensors)})'
    )
    data_line = handle_fused_act(opt.FusedActivationFunction(), data_line)
    line = f'{output_name(op, 0, transform_tensors)} = {data_line}'

    return line


def parse_transposeconv2d(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.TransposeConvOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.TransposeConvOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    strides = [opt.StrideH(), opt.StrideW()]
    padding = "SAME" if opt.Padding() == tflite.Padding.SAME else "VALID"
    dilations = None

    data_line = (
        f'tf.nn.bias_add(tf.nn.conv2d_transpose({input_name(op, 2, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)}, output_shape={input_name(op, 0, transform_tensors)},'
        f' strides={strides}, padding="{padding}", dilations={dilations}), {input_name(op, 3, transform_tensors)})'
    )
    data_line = handle_fused_act(opt.FusedActivationFunction(), data_line)
    line = f'{output_name(op, 0, transform_tensors)} = {data_line}'

    return line


def parse_averagepool2d(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.Pool2DOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.Pool2DOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    strides = [opt.StrideH(), opt.StrideW()]
    padding = "SAME" if opt.Padding() == tflite.Padding.SAME else "VALID"
    ksize = [opt.FilterHeight(), opt.FilterWidth()]

    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.nn.avg_pool2d({input_name(op, 0, transform_tensors)},'
        f' strides={strides}, padding="{padding}", ksize={ksize})'
    )
    return line


def parse_fullyconnected(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.FullyConnectedOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.FullyConnectedOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    data_line = (
        f'tf.nn.bias_add(tf.matmul({input_name(op, 0, transform_tensors)}, {input_name(op, 1, transform_tensors)},'
        f' transpose_b=True), {input_name(op, 2, transform_tensors)})'
    )
    data_line = handle_fused_act(opt.FusedActivationFunction(), data_line)
    line = f'{output_name(op, 0, transform_tensors)} = {data_line}'

    return line


def parse_slice(op: tflite.Operator, transform_tensors: typing.Set[int]):
    assert op.BuiltinOptionsType() == tflite.BuiltinOptions.SliceOptions
    op_opt = op.BuiltinOptions()

    opt = tflite.SliceOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    line = (
        f'{output_name(op, 0, transform_tensors)} = tf.slice({input_name(op, 0, transform_tensors)},'
        f' {input_name(op, 1, transform_tensors)}, {input_name(op, 2, transform_tensors)})'
    )
    return line


# OP parser registration table
OP_PARSER_DICT = {
    'PAD': parse_pad,
    'ADD': parse_add,
    'RESIZE_BILINEAR': parse_resize,
    'RESHAPE': parse_reshape,
    'CONV_2D': parse_conv2d,
    'AVERAGE_POOL_2D': parse_averagepool2d,
    'DEPTHWISE_CONV_2D': parse_depthwiseconv2d,
    'FULLY_CONNECTED': parse_fullyconnected,
    'DEPTH_TO_SPACE': parse_depth2space,
    'TRANSPOSE_CONV': parse_transposeconv2d,
    'SLICE': parse_slice,
    'RELU': parse_relu,
}

# Header for the generated script
HEADER = """from tinynn.converter.utils.tflite import parse_model

import tensorflow as tf
import numpy as np

tfl_model = parse_model("{}")

"""

# Conversion logic for the generated script
CONVERT_LOGIC = """
tf_model = TFModel()

@tf.function
def wrapper({}):
    result = tf_model({})
    outputs = {}
    return outputs


tf.saved_model.save(tf_model,
    "saved_model",
    signatures=wrapper.get_concrete_function(
        {}
    ),
)
"""

## Constants for TF and TFLite mapping
# mapping between op code and op name
OP_NAME_MAPPING = {
    getattr(tflite.BuiltinOperator, k): k
    for k in dir(tflite.BuiltinOperator)
    if not k.startswith('__') and not k.endswith('__')
}

TFLITE_NP_TYPE_MAPPING = {
    tflite.TensorType.FLOAT32: "float32",
    tflite.TensorType.INT32: "int32",
    tflite.TensorType.BOOL: "bool",
    tflite.TensorType.FLOAT64: "float64",
    tflite.TensorType.INT64: "int64",
}

TFLITE_TF_TYPE_MAPPING = {
    tflite.TensorType.FLOAT32: "tf.float32",
    tflite.TensorType.INT32: "tf.int32",
    tflite.TensorType.BOOL: "tf.bool",
    tflite.TensorType.FLOAT64: "tf.float64",
    tflite.TensorType.INT64: "tf.int64",
}


## Main logic
def parse_tflite(path):
    if isinstance(path, str):
        model = parse_model(path)
    else:
        assert False, f"expected type str but got {type(path)}"

    assert model.SubgraphsLength() == 1, "Only one subgraph is supported"

    subgraph = model.Subgraphs(0)

    input_names = []
    input_shapes = []
    input_signatures = []

    output_file = open('generate_tf_savedmodel.py', 'w', encoding='utf-8')
    output_file.write(HEADER.format(os.path.abspath(path)))

    # Collect input info so that we can generate input signatures
    for i in range(subgraph.InputsLength()):
        inp = subgraph.Inputs(i)
        tensor = subgraph.Tensors(inp)

        dtype = TFLITE_TF_TYPE_MAPPING.get(tensor.Type(), None)
        if dtype is None:
            print('Dtype not supported:', tensor.Type())
            exit(1)

        input_names.append(f'tensor_{inp}')
        input_shapes.append(tuple(tensor.ShapeAsNumpy().tolist()))
        input_signatures.append(f"tf.TensorSpec(shape={input_shapes[-1]}, dtype={dtype})")

    line = '''class TFModel(tf.Module):
    def __init__(self):'''
    output_file.write(f'{line}\n')

    # For some ops, the layout of the weight data is different. We try to mark them here.
    buffer_transform_dict = {}
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode = model.OperatorCodes(op.OpcodeIndex())
        if opcode.BuiltinCode() in (tflite.BuiltinOperator.CONV_2D, tflite.BuiltinOperator.DEPTHWISE_CONV_2D):
            buffer_transform_dict[op.Inputs(1)] = 1
        elif opcode.BuiltinCode() == tflite.BuiltinOperator.TRANSPOSE_CONV:
            buffer_transform_dict[op.Inputs(1)] = 2

    # Generate buffer definitions in TF
    transform_tensors = set()
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        if tensor.Buffer() != 0:
            buffer = tensor.Buffer()

            dtype = TFLITE_NP_TYPE_MAPPING.get(tensor.Type(), None)
            if dtype is None:
                print('Dtype not supported:', tensor.Type())
                exit(1)

            shape = tensor.ShapeAsNumpy()

            # For marked weights, certain weight transformation is performed
            transform = buffer_transform_dict.get(i, None)
            data_line = (
                f'np.frombuffer(tfl_model.Buffers({buffer}).DataAsNumpy().tobytes(),'
                f' dtype="{dtype}").reshape({shape.tolist()})'
            )
            if transform in (1, 2):
                if transform == 1:
                    sequence = (1, 2, 3, 0)
                else:
                    sequence = (1, 2, 0, 3)
                data_line = f'np.transpose({data_line}, {sequence})'
            elif transform is not None:
                print(f'Unknown transform: {transform}')
                exit(1)

            line = f'self.tensor_{i} = tf.constant({data_line})'
            output_file.write(f'        {line}\n')
        else:
            transform_tensors.add(i)

    output_file.write('\n')

    line = f'''
    @tf.function
    def __call__(self, {", ".join(input_names)}):'''
    output_file.write(f'{line}\n')

    # Parsing operations and generating function calls in TF
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode = model.OperatorCodes(op.OpcodeIndex())
        code = opcode.BuiltinCode()

        if OP_NAME_MAPPING[code] in OP_PARSER_DICT:
            parser = OP_PARSER_DICT[OP_NAME_MAPPING[code]]
            line = parser(op, transform_tensors)
            output_file.write(f'        {line}\n')
        else:
            print(f'OpCode Unknown: {code}({OP_NAME_MAPPING[code]})')
            exit(1)

    # Generating return values according to model outputs
    output_names = []
    for i in range(subgraph.OutputsLength()):
        outp = subgraph.Outputs(i)

        output_names.append(f'tensor_{outp}')

    input_names_for_conversion = ", ".join((f'input_{i}' for i in range(len(input_names))))
    if len(output_names) == 1:
        output_for_conversion = '{"output_0": result}'
    else:
        outputs = [f'"output_{i}": result[i]' for i in range(len(output_names))]
        output_for_conversion = f'{{ {", ".join(outputs)} }}'
    input_signatures_for_conversion = ', '.join([f'input_{i}={val}' for i, val in enumerate(input_signatures)])

    line = f'return {", ".join(output_names)}'
    output_file.write(f'        {line}\n')
    conversion_logic = CONVERT_LOGIC.format(
        input_names_for_conversion, input_names_for_conversion, output_for_conversion, input_signatures_for_conversion
    )
    output_file.write(f'\n{conversion_logic}\n')

    output_file.close()


if __name__ == '__main__':
    parse_tflite('/workspaces/TinyNeuralNetwork/examples/converter/out/mbv1_224.tflite')
