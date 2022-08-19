import flatbuffers

from ...schemas.tflite import schema_generated as tflite
from . import BaseOperator, Offset, create_byte_array, create_numpy_array


class AddOperator(BaseOperator):
    def __init__(
        self, inputs, outputs, fusedActivationFunction=tflite.ActivationFunctionType.NONE, potScaleInt16=False
    ) -> None:
        super().__init__(tflite.BuiltinOperator.ADD, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.potScaleInt16 = potScaleInt16

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.AddOptionsStart(builder)
        tflite.AddOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.AddOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        options = tflite.AddOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.AddOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class AveragePool2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideW=0,
        strideH=0,
        filterWidth=0,
        filterHeight=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.AVERAGE_POOL_2D, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Pool2DOptionsStart(builder)
        tflite.Pool2DOptionsAddPadding(builder, self.padding)
        tflite.Pool2DOptionsAddStrideW(builder, self.strideW)
        tflite.Pool2DOptionsAddStrideH(builder, self.strideH)
        tflite.Pool2DOptionsAddFilterWidth(builder, self.filterWidth)
        tflite.Pool2DOptionsAddFilterHeight(builder, self.filterHeight)
        tflite.Pool2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.Pool2DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Pool2DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ConcatenationOperator(BaseOperator):
    def __init__(self, inputs, outputs, axis=0, fusedActivationFunction=tflite.ActivationFunctionType.NONE) -> None:
        super().__init__(tflite.BuiltinOperator.CONCATENATION, inputs, outputs)
        self.axis = axis
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ConcatenationOptionsStart(builder)
        tflite.ConcatenationOptionsAddAxis(builder, self.axis)
        tflite.ConcatenationOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.ConcatenationOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ConcatenationOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class Conv2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideW=0,
        strideH=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        dilationWFactor=0,
        dilationHFactor=0,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.CONV_2D, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.fusedActivationFunction = fusedActivationFunction
        self.dilationWFactor = dilationWFactor
        self.dilationHFactor = dilationHFactor

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Conv2DOptionsStart(builder)
        tflite.Conv2DOptionsAddPadding(builder, self.padding)
        tflite.Conv2DOptionsAddStrideW(builder, self.strideW)
        tflite.Conv2DOptionsAddStrideH(builder, self.strideH)
        tflite.Conv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.Conv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        tflite.Conv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        options = tflite.Conv2DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Conv2DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DepthwiseConv2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideW=0,
        strideH=0,
        depthMultiplier=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        dilationWFactor=0,
        dilationHFactor=0,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.DEPTHWISE_CONV_2D, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.depthMultiplier = depthMultiplier
        self.fusedActivationFunction = fusedActivationFunction
        self.dilationWFactor = dilationWFactor
        self.dilationHFactor = dilationHFactor

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.DepthwiseConv2DOptionsStart(builder)
        tflite.DepthwiseConv2DOptionsAddPadding(builder, self.padding)
        tflite.DepthwiseConv2DOptionsAddStrideW(builder, self.strideW)
        tflite.DepthwiseConv2DOptionsAddStrideH(builder, self.strideH)
        tflite.DepthwiseConv2DOptionsAddDepthMultiplier(builder, self.depthMultiplier)
        tflite.DepthwiseConv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.DepthwiseConv2DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        tflite.DepthwiseConv2DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        options = tflite.DepthwiseConv2DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.DepthwiseConv2DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DepthToSpaceOperator(BaseOperator):
    def __init__(self, inputs, outputs, blockSize=0) -> None:
        super().__init__(tflite.BuiltinOperator.DEPTH_TO_SPACE, inputs, outputs)
        self.blockSize = blockSize

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.DepthToSpaceOptionsStart(builder)
        tflite.DepthToSpaceOptionsAddBlockSize(builder, self.blockSize)
        options = tflite.DepthToSpaceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.DepthToSpaceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DequantizeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.DEQUANTIZE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.DequantizeOptionsStart(builder)
        options = tflite.DequantizeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.DequantizeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class EmbeddingLookupOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.EMBEDDING_LOOKUP, inputs, outputs)


class FloorOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.FLOOR, inputs, outputs)


class FullyConnectedOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        weightsFormat=tflite.FullyConnectedOptionsWeightsFormat.DEFAULT,
        keepNumDims=False,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.FULLY_CONNECTED, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.weightsFormat = weightsFormat
        self.keepNumDims = keepNumDims
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.FullyConnectedOptionsStart(builder)
        tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.FullyConnectedOptionsAddWeightsFormat(builder, self.weightsFormat)
        tflite.FullyConnectedOptionsAddKeepNumDims(builder, self.keepNumDims)
        tflite.FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.FullyConnectedOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FullyConnectedOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class HashtableLookupOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.HASHTABLE_LOOKUP, inputs, outputs)


class L2NormalizationOperator(BaseOperator):
    def __init__(self, inputs, outputs, fusedActivationFunction=tflite.ActivationFunctionType.NONE) -> None:
        super().__init__(tflite.BuiltinOperator.L2_NORMALIZATION, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.L2NormOptionsStart(builder)
        tflite.L2NormOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.L2NormOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.L2NormOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class L2Pool2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideW=0,
        strideH=0,
        filterWidth=0,
        filterHeight=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.L2_POOL_2D, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Pool2DOptionsStart(builder)
        tflite.Pool2DOptionsAddPadding(builder, self.padding)
        tflite.Pool2DOptionsAddStrideW(builder, self.strideW)
        tflite.Pool2DOptionsAddStrideH(builder, self.strideH)
        tflite.Pool2DOptionsAddFilterWidth(builder, self.filterWidth)
        tflite.Pool2DOptionsAddFilterHeight(builder, self.filterHeight)
        tflite.Pool2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.Pool2DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Pool2DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LocalResponseNormalizationOperator(BaseOperator):
    def __init__(self, inputs, outputs, radius=0, bias=0.0, alpha=0.0, beta=0.0) -> None:
        super().__init__(tflite.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION, inputs, outputs)
        self.radius = radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LocalResponseNormalizationOptionsStart(builder)
        tflite.LocalResponseNormalizationOptionsAddRadius(builder, self.radius)
        tflite.LocalResponseNormalizationOptionsAddBias(builder, self.bias)
        tflite.LocalResponseNormalizationOptionsAddAlpha(builder, self.alpha)
        tflite.LocalResponseNormalizationOptionsAddBeta(builder, self.beta)
        options = tflite.LocalResponseNormalizationOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LocalResponseNormalizationOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogisticOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.LOGISTIC, inputs, outputs)


class LshProjectionOperator(BaseOperator):
    def __init__(self, inputs, outputs, type=tflite.LSHProjectionType.UNKNOWN) -> None:
        super().__init__(tflite.BuiltinOperator.LSH_PROJECTION, inputs, outputs)
        self.type = type

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LSHProjectionOptionsStart(builder)
        tflite.LSHProjectionOptionsAddType(builder, self.type)
        options = tflite.LSHProjectionOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LSHProjectionOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LstmOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        cellClip=0.0,
        projClip=0.0,
        kernelType=tflite.LSTMKernelType.FULL,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LSTM, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.cellClip = cellClip
        self.projClip = projClip
        self.kernelType = kernelType
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LSTMOptionsStart(builder)
        tflite.LSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.LSTMOptionsAddCellClip(builder, self.cellClip)
        tflite.LSTMOptionsAddProjClip(builder, self.projClip)
        tflite.LSTMOptionsAddKernelType(builder, self.kernelType)
        tflite.LSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.LSTMOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LSTMOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MaxPool2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideW=0,
        strideH=0,
        filterWidth=0,
        filterHeight=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.MAX_POOL_2D, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Pool2DOptionsStart(builder)
        tflite.Pool2DOptionsAddPadding(builder, self.padding)
        tflite.Pool2DOptionsAddStrideW(builder, self.strideW)
        tflite.Pool2DOptionsAddStrideH(builder, self.strideH)
        tflite.Pool2DOptionsAddFilterWidth(builder, self.filterWidth)
        tflite.Pool2DOptionsAddFilterHeight(builder, self.filterHeight)
        tflite.Pool2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.Pool2DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Pool2DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MulOperator(BaseOperator):
    def __init__(self, inputs, outputs, fusedActivationFunction=tflite.ActivationFunctionType.NONE) -> None:
        super().__init__(tflite.BuiltinOperator.MUL, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MulOptionsStart(builder)
        tflite.MulOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.MulOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MulOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReluOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.RELU, inputs, outputs)


class ReluN1To1Operator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.RELU_N1_TO_1, inputs, outputs)


class Relu6Operator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.RELU6, inputs, outputs)


class ReshapeOperator(BaseOperator):
    def __init__(self, inputs, outputs, newShape=()) -> None:
        super().__init__(tflite.BuiltinOperator.RESHAPE, inputs, outputs)
        self.newShape = newShape

    def build(self, builder: flatbuffers.Builder) -> Offset:
        newShape = create_numpy_array(builder, tflite.ReshapeOptions.NewShape, self.newShape)
        tflite.ReshapeOptionsStart(builder)
        tflite.ReshapeOptionsAddNewShape(builder, newShape)
        options = tflite.ReshapeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReshapeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ResizeBilinearOperator(BaseOperator):
    def __init__(self, inputs, outputs, alignCorners=False, halfPixelCenters=False) -> None:
        super().__init__(tflite.BuiltinOperator.RESIZE_BILINEAR, inputs, outputs)
        self.alignCorners = alignCorners
        self.halfPixelCenters = halfPixelCenters

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ResizeBilinearOptionsStart(builder)
        tflite.ResizeBilinearOptionsAddAlignCorners(builder, self.alignCorners)
        tflite.ResizeBilinearOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        options = tflite.ResizeBilinearOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ResizeBilinearOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class RnnOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.RNN, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RNNOptionsStart(builder)
        tflite.RNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.RNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.RNNOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RNNOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SoftmaxOperator(BaseOperator):
    def __init__(self, inputs, outputs, beta=0.0) -> None:
        super().__init__(tflite.BuiltinOperator.SOFTMAX, inputs, outputs)
        self.beta = beta

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SoftmaxOptionsStart(builder)
        tflite.SoftmaxOptionsAddBeta(builder, self.beta)
        options = tflite.SoftmaxOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SoftmaxOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SpaceToDepthOperator(BaseOperator):
    def __init__(self, inputs, outputs, blockSize=0) -> None:
        super().__init__(tflite.BuiltinOperator.SPACE_TO_DEPTH, inputs, outputs)
        self.blockSize = blockSize

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SpaceToDepthOptionsStart(builder)
        tflite.SpaceToDepthOptionsAddBlockSize(builder, self.blockSize)
        options = tflite.SpaceToDepthOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SpaceToDepthOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SvdfOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        rank=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SVDF, inputs, outputs)
        self.rank = rank
        self.fusedActivationFunction = fusedActivationFunction
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SVDFOptionsStart(builder)
        tflite.SVDFOptionsAddRank(builder, self.rank)
        tflite.SVDFOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.SVDFOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.SVDFOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SVDFOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class TanhOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.TANH, inputs, outputs)


class ConcatEmbeddingsOperator(BaseOperator):
    def __init__(self, inputs, outputs, numChannels=0, numColumnsPerChannel=(), embeddingDimPerChannel=()) -> None:
        super().__init__(tflite.BuiltinOperator.CONCAT_EMBEDDINGS, inputs, outputs)
        self.numChannels = numChannels
        self.numColumnsPerChannel = numColumnsPerChannel
        self.embeddingDimPerChannel = embeddingDimPerChannel

    def build(self, builder: flatbuffers.Builder) -> Offset:
        numColumnsPerChannel = create_numpy_array(
            builder, tflite.ConcatEmbeddingsOptions.NumColumnsPerChannel, self.numColumnsPerChannel
        )
        embeddingDimPerChannel = create_numpy_array(
            builder, tflite.ConcatEmbeddingsOptions.EmbeddingDimPerChannel, self.embeddingDimPerChannel
        )
        tflite.ConcatEmbeddingsOptionsStart(builder)
        tflite.ConcatEmbeddingsOptionsAddNumChannels(builder, self.numChannels)
        tflite.ConcatEmbeddingsOptionsAddNumColumnsPerChannel(builder, numColumnsPerChannel)
        tflite.ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(builder, embeddingDimPerChannel)
        options = tflite.ConcatEmbeddingsOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ConcatEmbeddingsOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SkipGramOperator(BaseOperator):
    def __init__(self, inputs, outputs, ngramSize=0, maxSkipSize=0, includeAllNgrams=False) -> None:
        super().__init__(tflite.BuiltinOperator.SKIP_GRAM, inputs, outputs)
        self.ngramSize = ngramSize
        self.maxSkipSize = maxSkipSize
        self.includeAllNgrams = includeAllNgrams

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SkipGramOptionsStart(builder)
        tflite.SkipGramOptionsAddNgramSize(builder, self.ngramSize)
        tflite.SkipGramOptionsAddMaxSkipSize(builder, self.maxSkipSize)
        tflite.SkipGramOptionsAddIncludeAllNgrams(builder, self.includeAllNgrams)
        options = tflite.SkipGramOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SkipGramOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CallOperator(BaseOperator):
    def __init__(self, inputs, outputs, subgraph=0) -> None:
        super().__init__(tflite.BuiltinOperator.CALL, inputs, outputs)
        self.subgraph = subgraph

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.CallOptionsStart(builder)
        tflite.CallOptionsAddSubgraph(builder, self.subgraph)
        options = tflite.CallOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.CallOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CustomOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.CUSTOM, inputs, outputs)
        self.custom_options = None

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        if self.custom_options is not None:
            custom_options = create_byte_array(builder, tflite.Operator.CustomOptions, self.custom_options)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.NONE)
        tflite.OperatorAddBuiltinOptions(builder, 0)

        if self.custom_options is not None:
            tflite.OperatorAddCustomOptionsFormat(builder, tflite.CustomOptionsFormat.FLEXBUFFERS)
            tflite.OperatorAddCustomOptions(builder, custom_options)

        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class EmbeddingLookupSparseOperator(BaseOperator):
    def __init__(self, inputs, outputs, combiner=tflite.CombinerType.SUM) -> None:
        super().__init__(tflite.BuiltinOperator.EMBEDDING_LOOKUP_SPARSE, inputs, outputs)
        self.combiner = combiner

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.EmbeddingLookupSparseOptionsStart(builder)
        tflite.EmbeddingLookupSparseOptionsAddCombiner(builder, self.combiner)
        options = tflite.EmbeddingLookupSparseOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.EmbeddingLookupSparseOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class PadOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.PAD, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.PadOptionsStart(builder)
        options = tflite.PadOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.PadOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class UnidirectionalSequenceRnnOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        timeMajor=False,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN, inputs, outputs)
        self.timeMajor = timeMajor
        self.fusedActivationFunction = fusedActivationFunction
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SequenceRNNOptionsStart(builder)
        tflite.SequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        tflite.SequenceRNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.SequenceRNNOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SequenceRNNOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class GatherOperator(BaseOperator):
    def __init__(self, inputs, outputs, axis=0, batchDims=0) -> None:
        super().__init__(tflite.BuiltinOperator.GATHER, inputs, outputs)
        self.axis = axis
        self.batchDims = batchDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.GatherOptionsStart(builder)
        tflite.GatherOptionsAddAxis(builder, self.axis)
        tflite.GatherOptionsAddBatchDims(builder, self.batchDims)
        options = tflite.GatherOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.GatherOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BatchToSpaceNdOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.BATCH_TO_SPACE_ND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.BatchToSpaceNDOptionsStart(builder)
        options = tflite.BatchToSpaceNDOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BatchToSpaceNDOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SpaceToBatchNdOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SPACE_TO_BATCH_ND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SpaceToBatchNDOptionsStart(builder)
        options = tflite.SpaceToBatchNDOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SpaceToBatchNDOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class TransposeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.TRANSPOSE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.TransposeOptionsStart(builder)
        options = tflite.TransposeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.TransposeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MeanOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.MEAN, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SubOperator(BaseOperator):
    def __init__(
        self, inputs, outputs, fusedActivationFunction=tflite.ActivationFunctionType.NONE, potScaleInt16=False
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SUB, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.potScaleInt16 = potScaleInt16

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SubOptionsStart(builder)
        tflite.SubOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.SubOptionsAddPotScaleInt16(builder, self.potScaleInt16)
        options = tflite.SubOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SubOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DivOperator(BaseOperator):
    def __init__(self, inputs, outputs, fusedActivationFunction=tflite.ActivationFunctionType.NONE) -> None:
        super().__init__(tflite.BuiltinOperator.DIV, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.DivOptionsStart(builder)
        tflite.DivOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        options = tflite.DivOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.DivOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SqueezeOperator(BaseOperator):
    def __init__(self, inputs, outputs, squeezeDims=()) -> None:
        super().__init__(tflite.BuiltinOperator.SQUEEZE, inputs, outputs)
        self.squeezeDims = squeezeDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        squeezeDims = create_numpy_array(builder, tflite.SqueezeOptions.SqueezeDims, self.squeezeDims)
        tflite.SqueezeOptionsStart(builder)
        tflite.SqueezeOptionsAddSqueezeDims(builder, squeezeDims)
        options = tflite.SqueezeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SqueezeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class UnidirectionalSequenceLstmOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        cellClip=0.0,
        projClip=0.0,
        timeMajor=False,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.cellClip = cellClip
        self.projClip = projClip
        self.timeMajor = timeMajor
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.UnidirectionalSequenceLSTMOptionsStart(builder)
        tflite.UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.UnidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        tflite.UnidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        tflite.UnidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        tflite.UnidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.UnidirectionalSequenceLSTMOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.UnidirectionalSequenceLSTMOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class StridedSliceOperator(BaseOperator):
    def __init__(
        self, inputs, outputs, beginMask=0, endMask=0, ellipsisMask=0, newAxisMask=0, shrinkAxisMask=0
    ) -> None:
        super().__init__(tflite.BuiltinOperator.STRIDED_SLICE, inputs, outputs)
        self.beginMask = beginMask
        self.endMask = endMask
        self.ellipsisMask = ellipsisMask
        self.newAxisMask = newAxisMask
        self.shrinkAxisMask = shrinkAxisMask

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.StridedSliceOptionsStart(builder)
        tflite.StridedSliceOptionsAddBeginMask(builder, self.beginMask)
        tflite.StridedSliceOptionsAddEndMask(builder, self.endMask)
        tflite.StridedSliceOptionsAddEllipsisMask(builder, self.ellipsisMask)
        tflite.StridedSliceOptionsAddNewAxisMask(builder, self.newAxisMask)
        tflite.StridedSliceOptionsAddShrinkAxisMask(builder, self.shrinkAxisMask)
        options = tflite.StridedSliceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.StridedSliceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BidirectionalSequenceRnnOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        timeMajor=False,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        mergeOutputs=False,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN, inputs, outputs)
        self.timeMajor = timeMajor
        self.fusedActivationFunction = fusedActivationFunction
        self.mergeOutputs = mergeOutputs
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.BidirectionalSequenceRNNOptionsStart(builder)
        tflite.BidirectionalSequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        tflite.BidirectionalSequenceRNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.BidirectionalSequenceRNNOptionsAddMergeOutputs(builder, self.mergeOutputs)
        tflite.BidirectionalSequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.BidirectionalSequenceRNNOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BidirectionalSequenceRNNOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ExpOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.EXP, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ExpOptionsStart(builder)
        options = tflite.ExpOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ExpOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class TopkV2Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.TOPK_V2, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.TopKV2OptionsStart(builder)
        options = tflite.TopKV2OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.TopKV2Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SplitOperator(BaseOperator):
    def __init__(self, inputs, outputs, numSplits=0) -> None:
        super().__init__(tflite.BuiltinOperator.SPLIT, inputs, outputs)
        self.numSplits = numSplits

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SplitOptionsStart(builder)
        tflite.SplitOptionsAddNumSplits(builder, self.numSplits)
        options = tflite.SplitOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SplitOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogSoftmaxOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LOG_SOFTMAX, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LogSoftmaxOptionsStart(builder)
        options = tflite.LogSoftmaxOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LogSoftmaxOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DelegateOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.DELEGATE, inputs, outputs)


class BidirectionalSequenceLstmOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        cellClip=0.0,
        projClip=0.0,
        mergeOutputs=False,
        timeMajor=False,
        asymmetricQuantizeInputs=False,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM, inputs, outputs)
        self.fusedActivationFunction = fusedActivationFunction
        self.cellClip = cellClip
        self.projClip = projClip
        self.mergeOutputs = mergeOutputs
        self.timeMajor = timeMajor
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.BidirectionalSequenceLSTMOptionsStart(builder)
        tflite.BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.BidirectionalSequenceLSTMOptionsAddCellClip(builder, self.cellClip)
        tflite.BidirectionalSequenceLSTMOptionsAddProjClip(builder, self.projClip)
        tflite.BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, self.mergeOutputs)
        tflite.BidirectionalSequenceLSTMOptionsAddTimeMajor(builder, self.timeMajor)
        tflite.BidirectionalSequenceLSTMOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.BidirectionalSequenceLSTMOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BidirectionalSequenceLSTMOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CastOperator(BaseOperator):
    def __init__(
        self, inputs, outputs, inDataType=tflite.TensorType.FLOAT32, outDataType=tflite.TensorType.FLOAT32
    ) -> None:
        super().__init__(tflite.BuiltinOperator.CAST, inputs, outputs)
        self.inDataType = inDataType
        self.outDataType = outDataType

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.CastOptionsStart(builder)
        tflite.CastOptionsAddInDataType(builder, self.inDataType)
        tflite.CastOptionsAddOutDataType(builder, self.outDataType)
        options = tflite.CastOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.CastOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class PreluOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.PRELU, inputs, outputs)


class MaximumOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.MAXIMUM, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MaximumMinimumOptionsStart(builder)
        options = tflite.MaximumMinimumOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MaximumMinimumOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ArgMaxOperator(BaseOperator):
    def __init__(self, inputs, outputs, outputType=tflite.TensorType.FLOAT32) -> None:
        super().__init__(tflite.BuiltinOperator.ARG_MAX, inputs, outputs)
        self.outputType = outputType

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ArgMaxOptionsStart(builder)
        tflite.ArgMaxOptionsAddOutputType(builder, self.outputType)
        options = tflite.ArgMaxOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ArgMaxOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MinimumOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.MINIMUM, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MaximumMinimumOptionsStart(builder)
        options = tflite.MaximumMinimumOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MaximumMinimumOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LessOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LESS, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LessOptionsStart(builder)
        options = tflite.LessOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LessOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class NegOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.NEG, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.NegOptionsStart(builder)
        options = tflite.NegOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.NegOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class Padv2Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.PADV2, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.PadV2OptionsStart(builder)
        options = tflite.PadV2OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.PadV2Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class GreaterOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.GREATER, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.GreaterOptionsStart(builder)
        options = tflite.GreaterOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.GreaterOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class GreaterEqualOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.GREATER_EQUAL, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.GreaterEqualOptionsStart(builder)
        options = tflite.GreaterEqualOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.GreaterEqualOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LessEqualOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LESS_EQUAL, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LessEqualOptionsStart(builder)
        options = tflite.LessEqualOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LessEqualOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SelectOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SELECT, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SelectOptionsStart(builder)
        options = tflite.SelectOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SelectOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SliceOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SLICE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SliceOptionsStart(builder)
        options = tflite.SliceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SliceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SinOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.SIN, inputs, outputs)


class TransposeConvOperator(BaseOperator):
    def __init__(self, inputs, outputs, padding=tflite.Padding.SAME, strideW=0, strideH=0) -> None:
        super().__init__(tflite.BuiltinOperator.TRANSPOSE_CONV, inputs, outputs)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.TransposeConvOptionsStart(builder)
        tflite.TransposeConvOptionsAddPadding(builder, self.padding)
        tflite.TransposeConvOptionsAddStrideW(builder, self.strideW)
        tflite.TransposeConvOptionsAddStrideH(builder, self.strideH)
        options = tflite.TransposeConvOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.TransposeConvOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SparseToDenseOperator(BaseOperator):
    def __init__(self, inputs, outputs, validateIndices=False) -> None:
        super().__init__(tflite.BuiltinOperator.SPARSE_TO_DENSE, inputs, outputs)
        self.validateIndices = validateIndices

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SparseToDenseOptionsStart(builder)
        tflite.SparseToDenseOptionsAddValidateIndices(builder, self.validateIndices)
        options = tflite.SparseToDenseOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SparseToDenseOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class TileOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.TILE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.TileOptionsStart(builder)
        options = tflite.TileOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.TileOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ExpandDimsOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.EXPAND_DIMS, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ExpandDimsOptionsStart(builder)
        options = tflite.ExpandDimsOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ExpandDimsOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class EqualOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.EQUAL, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.EqualOptionsStart(builder)
        options = tflite.EqualOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.EqualOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class NotEqualOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.NOT_EQUAL, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.NotEqualOptionsStart(builder)
        options = tflite.NotEqualOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.NotEqualOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.LOG, inputs, outputs)


class SumOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.SUM, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SqrtOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.SQRT, inputs, outputs)


class RsqrtOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.RSQRT, inputs, outputs)


class ShapeOperator(BaseOperator):
    def __init__(self, inputs, outputs, outType=tflite.TensorType.FLOAT32) -> None:
        super().__init__(tflite.BuiltinOperator.SHAPE, inputs, outputs)
        self.outType = outType

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ShapeOptionsStart(builder)
        tflite.ShapeOptionsAddOutType(builder, self.outType)
        options = tflite.ShapeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ShapeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class PowOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.POW, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.PowOptionsStart(builder)
        options = tflite.PowOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.PowOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ArgMinOperator(BaseOperator):
    def __init__(self, inputs, outputs, outputType=tflite.TensorType.FLOAT32) -> None:
        super().__init__(tflite.BuiltinOperator.ARG_MIN, inputs, outputs)
        self.outputType = outputType

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ArgMinOptionsStart(builder)
        tflite.ArgMinOptionsAddOutputType(builder, self.outputType)
        options = tflite.ArgMinOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ArgMinOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class FakeQuantOperator(BaseOperator):
    def __init__(self, inputs, outputs, min=0.0, max=0.0, numBits=0, narrowRange=False) -> None:
        super().__init__(tflite.BuiltinOperator.FAKE_QUANT, inputs, outputs)
        self.min = min
        self.max = max
        self.numBits = numBits
        self.narrowRange = narrowRange

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.FakeQuantOptionsStart(builder)
        tflite.FakeQuantOptionsAddMin(builder, self.min)
        tflite.FakeQuantOptionsAddMax(builder, self.max)
        tflite.FakeQuantOptionsAddNumBits(builder, self.numBits)
        tflite.FakeQuantOptionsAddNarrowRange(builder, self.narrowRange)
        options = tflite.FakeQuantOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FakeQuantOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReduceProdOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.REDUCE_PROD, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReduceMaxOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.REDUCE_MAX, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class PackOperator(BaseOperator):
    def __init__(self, inputs, outputs, valuesCount=0, axis=0) -> None:
        super().__init__(tflite.BuiltinOperator.PACK, inputs, outputs)
        self.valuesCount = valuesCount
        self.axis = axis

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.PackOptionsStart(builder)
        tflite.PackOptionsAddValuesCount(builder, self.valuesCount)
        tflite.PackOptionsAddAxis(builder, self.axis)
        options = tflite.PackOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.PackOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogicalOrOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LOGICAL_OR, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LogicalOrOptionsStart(builder)
        options = tflite.LogicalOrOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LogicalOrOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class OneHotOperator(BaseOperator):
    def __init__(self, inputs, outputs, axis=0) -> None:
        super().__init__(tflite.BuiltinOperator.ONE_HOT, inputs, outputs)
        self.axis = axis

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.OneHotOptionsStart(builder)
        tflite.OneHotOptionsAddAxis(builder, self.axis)
        options = tflite.OneHotOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.OneHotOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogicalAndOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LOGICAL_AND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LogicalAndOptionsStart(builder)
        options = tflite.LogicalAndOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LogicalAndOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LogicalNotOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.LOGICAL_NOT, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LogicalNotOptionsStart(builder)
        options = tflite.LogicalNotOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LogicalNotOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class UnpackOperator(BaseOperator):
    def __init__(self, inputs, outputs, num=0, axis=0) -> None:
        super().__init__(tflite.BuiltinOperator.UNPACK, inputs, outputs)
        self.num = num
        self.axis = axis

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.UnpackOptionsStart(builder)
        tflite.UnpackOptionsAddNum(builder, self.num)
        tflite.UnpackOptionsAddAxis(builder, self.axis)
        options = tflite.UnpackOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.UnpackOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReduceMinOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.REDUCE_MIN, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class FloorDivOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.FLOOR_DIV, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.FloorDivOptionsStart(builder)
        options = tflite.FloorDivOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FloorDivOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReduceAnyOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.REDUCE_ANY, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SquareOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SQUARE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SquareOptionsStart(builder)
        options = tflite.SquareOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SquareOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ZerosLikeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.ZEROS_LIKE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ZerosLikeOptionsStart(builder)
        options = tflite.ZerosLikeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ZerosLikeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class FillOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.FILL, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.FillOptionsStart(builder)
        options = tflite.FillOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FillOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class FloorModOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.FLOOR_MOD, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.FloorModOptionsStart(builder)
        options = tflite.FloorModOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FloorModOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class RangeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.RANGE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RangeOptionsStart(builder)
        options = tflite.RangeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RangeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ResizeNearestNeighborOperator(BaseOperator):
    def __init__(self, inputs, outputs, alignCorners=False, halfPixelCenters=False) -> None:
        super().__init__(tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, inputs, outputs)
        self.alignCorners = alignCorners
        self.halfPixelCenters = halfPixelCenters

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ResizeNearestNeighborOptionsStart(builder)
        tflite.ResizeNearestNeighborOptionsAddAlignCorners(builder, self.alignCorners)
        tflite.ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, self.halfPixelCenters)
        options = tflite.ResizeNearestNeighborOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ResizeNearestNeighborOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class LeakyReluOperator(BaseOperator):
    def __init__(self, inputs, outputs, alpha=0.0) -> None:
        super().__init__(tflite.BuiltinOperator.LEAKY_RELU, inputs, outputs)
        self.alpha = alpha

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.LeakyReluOptionsStart(builder)
        tflite.LeakyReluOptionsAddAlpha(builder, self.alpha)
        options = tflite.LeakyReluOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.LeakyReluOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SquaredDifferenceOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SQUARED_DIFFERENCE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SquaredDifferenceOptionsStart(builder)
        options = tflite.SquaredDifferenceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SquaredDifferenceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MirrorPadOperator(BaseOperator):
    def __init__(self, inputs, outputs, mode=tflite.MirrorPadMode.REFLECT) -> None:
        super().__init__(tflite.BuiltinOperator.MIRROR_PAD, inputs, outputs)
        self.mode = mode

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MirrorPadOptionsStart(builder)
        tflite.MirrorPadOptionsAddMode(builder, self.mode)
        options = tflite.MirrorPadOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MirrorPadOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class AbsOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.ABS, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.AbsOptionsStart(builder)
        options = tflite.AbsOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.AbsOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SplitVOperator(BaseOperator):
    def __init__(self, inputs, outputs, numSplits=0) -> None:
        super().__init__(tflite.BuiltinOperator.SPLIT_V, inputs, outputs)
        self.numSplits = numSplits

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SplitVOptionsStart(builder)
        tflite.SplitVOptionsAddNumSplits(builder, self.numSplits)
        options = tflite.SplitVOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SplitVOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class UniqueOperator(BaseOperator):
    def __init__(self, inputs, outputs, idxOutType=tflite.TensorType.FLOAT32) -> None:
        super().__init__(tflite.BuiltinOperator.UNIQUE, inputs, outputs)
        self.idxOutType = idxOutType

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.UniqueOptionsStart(builder)
        tflite.UniqueOptionsAddIdxOutType(builder, self.idxOutType)
        options = tflite.UniqueOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.UniqueOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CeilOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.CEIL, inputs, outputs)


class ReverseV2Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.REVERSE_V2, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReverseV2OptionsStart(builder)
        options = tflite.ReverseV2OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReverseV2Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class AddNOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.ADD_N, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.AddNOptionsStart(builder)
        options = tflite.AddNOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.AddNOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class GatherNdOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.GATHER_ND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.GatherNdOptionsStart(builder)
        options = tflite.GatherNdOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.GatherNdOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CosOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.COS, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.CosOptionsStart(builder)
        options = tflite.CosOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.CosOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class WhereOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.WHERE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.WhereOptionsStart(builder)
        options = tflite.WhereOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.WhereOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class RankOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.RANK, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RankOptionsStart(builder)
        options = tflite.RankOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RankOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class EluOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.ELU, inputs, outputs)


class ReverseSequenceOperator(BaseOperator):
    def __init__(self, inputs, outputs, seqDim=0, batchDim=0) -> None:
        super().__init__(tflite.BuiltinOperator.REVERSE_SEQUENCE, inputs, outputs)
        self.seqDim = seqDim
        self.batchDim = batchDim

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReverseSequenceOptionsStart(builder)
        tflite.ReverseSequenceOptionsAddSeqDim(builder, self.seqDim)
        tflite.ReverseSequenceOptionsAddBatchDim(builder, self.batchDim)
        options = tflite.ReverseSequenceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReverseSequenceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MatrixDiagOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.MATRIX_DIAG, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MatrixDiagOptionsStart(builder)
        options = tflite.MatrixDiagOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MatrixDiagOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class QuantizeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.QUANTIZE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.QuantizeOptionsStart(builder)
        options = tflite.QuantizeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.QuantizeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MatrixSetDiagOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.MATRIX_SET_DIAG, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.MatrixSetDiagOptionsStart(builder)
        options = tflite.MatrixSetDiagOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.MatrixSetDiagOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class RoundOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.ROUND, inputs, outputs)


class HardSwishOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.HARD_SWISH, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.HardSwishOptionsStart(builder)
        options = tflite.HardSwishOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.HardSwishOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class IfOperator(BaseOperator):
    def __init__(self, inputs, outputs, thenSubgraphIndex=0, elseSubgraphIndex=0) -> None:
        super().__init__(tflite.BuiltinOperator.IF, inputs, outputs)
        self.thenSubgraphIndex = thenSubgraphIndex
        self.elseSubgraphIndex = elseSubgraphIndex

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.IfOptionsStart(builder)
        tflite.IfOptionsAddThenSubgraphIndex(builder, self.thenSubgraphIndex)
        tflite.IfOptionsAddElseSubgraphIndex(builder, self.elseSubgraphIndex)
        options = tflite.IfOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.IfOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class WhileOperator(BaseOperator):
    def __init__(self, inputs, outputs, condSubgraphIndex=0, bodySubgraphIndex=0) -> None:
        super().__init__(tflite.BuiltinOperator.WHILE, inputs, outputs)
        self.condSubgraphIndex = condSubgraphIndex
        self.bodySubgraphIndex = bodySubgraphIndex

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.WhileOptionsStart(builder)
        tflite.WhileOptionsAddCondSubgraphIndex(builder, self.condSubgraphIndex)
        tflite.WhileOptionsAddBodySubgraphIndex(builder, self.bodySubgraphIndex)
        options = tflite.WhileOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.WhileOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class NonMaxSuppressionV4Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V4, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.NonMaxSuppressionV4OptionsStart(builder)
        options = tflite.NonMaxSuppressionV4OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.NonMaxSuppressionV4Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class NonMaxSuppressionV5Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.NON_MAX_SUPPRESSION_V5, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.NonMaxSuppressionV5OptionsStart(builder)
        options = tflite.NonMaxSuppressionV5OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.NonMaxSuppressionV5Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ScatterNdOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SCATTER_ND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ScatterNdOptionsStart(builder)
        options = tflite.ScatterNdOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ScatterNdOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SelectV2Operator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SELECT_V2, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SelectV2OptionsStart(builder)
        options = tflite.SelectV2OptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SelectV2Options)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class DensifyOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.DENSIFY, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.DensifyOptionsStart(builder)
        options = tflite.DensifyOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.DensifyOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class SegmentSumOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.SEGMENT_SUM, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.SegmentSumOptionsStart(builder)
        options = tflite.SegmentSumOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.SegmentSumOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BatchMatmulOperator(BaseOperator):
    def __init__(self, inputs, outputs, adjX=False, adjY=False, asymmetricQuantizeInputs=False) -> None:
        super().__init__(tflite.BuiltinOperator.BATCH_MATMUL, inputs, outputs)
        self.adjX = adjX
        self.adjY = adjY
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.BatchMatMulOptionsStart(builder)
        tflite.BatchMatMulOptionsAddAdjX(builder, self.adjX)
        tflite.BatchMatMulOptionsAddAdjY(builder, self.adjY)
        tflite.BatchMatMulOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        options = tflite.BatchMatMulOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BatchMatMulOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class PlaceholderForGreaterOpCodesOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES, inputs, outputs)


class CumsumOperator(BaseOperator):
    def __init__(self, inputs, outputs, exclusive=False, reverse=False) -> None:
        super().__init__(tflite.BuiltinOperator.CUMSUM, inputs, outputs)
        self.exclusive = exclusive
        self.reverse = reverse

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.CumsumOptionsStart(builder)
        tflite.CumsumOptionsAddExclusive(builder, self.exclusive)
        tflite.CumsumOptionsAddReverse(builder, self.reverse)
        options = tflite.CumsumOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.CumsumOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class CallOnceOperator(BaseOperator):
    def __init__(self, inputs, outputs, initSubgraphIndex=0) -> None:
        super().__init__(tflite.BuiltinOperator.CALL_ONCE, inputs, outputs)
        self.initSubgraphIndex = initSubgraphIndex

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.CallOnceOptionsStart(builder)
        tflite.CallOnceOptionsAddInitSubgraphIndex(builder, self.initSubgraphIndex)
        options = tflite.CallOnceOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.CallOnceOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BroadcastToOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.BROADCAST_TO, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.BroadcastToOptionsStart(builder)
        options = tflite.BroadcastToOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BroadcastToOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class Rfft2dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.RFFT2D, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Rfft2dOptionsStart(builder)
        options = tflite.Rfft2dOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Rfft2dOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class Conv3dOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideD=0,
        strideW=0,
        strideH=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        dilationDFactor=0,
        dilationWFactor=0,
        dilationHFactor=0,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.CONV_3D, inputs, outputs)
        self.padding = padding
        self.strideD = strideD
        self.strideW = strideW
        self.strideH = strideH
        self.fusedActivationFunction = fusedActivationFunction
        self.dilationDFactor = dilationDFactor
        self.dilationWFactor = dilationWFactor
        self.dilationHFactor = dilationHFactor

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Conv3DOptionsStart(builder)
        tflite.Conv3DOptionsAddPadding(builder, self.padding)
        tflite.Conv3DOptionsAddStrideD(builder, self.strideD)
        tflite.Conv3DOptionsAddStrideW(builder, self.strideW)
        tflite.Conv3DOptionsAddStrideH(builder, self.strideH)
        tflite.Conv3DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.Conv3DOptionsAddDilationDFactor(builder, self.dilationDFactor)
        tflite.Conv3DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        tflite.Conv3DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        options = tflite.Conv3DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Conv3DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ImagOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.IMAG, inputs, outputs)


class RealOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.REAL, inputs, outputs)


class ComplexAbsOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.COMPLEX_ABS, inputs, outputs)


class HashtableOperator(BaseOperator):
    def __init__(
        self, inputs, outputs, tableId=0, keyDtype=tflite.TensorType.FLOAT32, valueDtype=tflite.TensorType.FLOAT32
    ) -> None:
        super().__init__(tflite.BuiltinOperator.HASHTABLE, inputs, outputs)
        self.tableId = tableId
        self.keyDtype = keyDtype
        self.valueDtype = valueDtype

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.HashtableOptionsStart(builder)
        tflite.HashtableOptionsAddTableId(builder, self.tableId)
        tflite.HashtableOptionsAddKeyDtype(builder, self.keyDtype)
        tflite.HashtableOptionsAddValueDtype(builder, self.valueDtype)
        options = tflite.HashtableOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.HashtableOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class HashtableFindOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.HASHTABLE_FIND, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.HashtableFindOptionsStart(builder)
        options = tflite.HashtableFindOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.HashtableFindOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class HashtableImportOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.HASHTABLE_IMPORT, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.HashtableImportOptionsStart(builder)
        options = tflite.HashtableImportOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.HashtableImportOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class HashtableSizeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.HASHTABLE_SIZE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.HashtableSizeOptionsStart(builder)
        options = tflite.HashtableSizeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.HashtableSizeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReduceAllOperator(BaseOperator):
    def __init__(self, inputs, outputs, keepDims=False) -> None:
        super().__init__(tflite.BuiltinOperator.REDUCE_ALL, inputs, outputs)
        self.keepDims = keepDims

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReducerOptionsStart(builder)
        tflite.ReducerOptionsAddKeepDims(builder, self.keepDims)
        options = tflite.ReducerOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReducerOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class Conv3dTransposeOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
        padding=tflite.Padding.SAME,
        strideD=0,
        strideW=0,
        strideH=0,
        fusedActivationFunction=tflite.ActivationFunctionType.NONE,
        dilationDFactor=0,
        dilationWFactor=0,
        dilationHFactor=0,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.CONV_3D_TRANSPOSE, inputs, outputs)
        self.padding = padding
        self.strideD = strideD
        self.strideW = strideW
        self.strideH = strideH
        self.fusedActivationFunction = fusedActivationFunction
        self.dilationDFactor = dilationDFactor
        self.dilationWFactor = dilationWFactor
        self.dilationHFactor = dilationHFactor

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.Conv3DOptionsStart(builder)
        tflite.Conv3DOptionsAddPadding(builder, self.padding)
        tflite.Conv3DOptionsAddStrideD(builder, self.strideD)
        tflite.Conv3DOptionsAddStrideW(builder, self.strideW)
        tflite.Conv3DOptionsAddStrideH(builder, self.strideH)
        tflite.Conv3DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        tflite.Conv3DOptionsAddDilationDFactor(builder, self.dilationDFactor)
        tflite.Conv3DOptionsAddDilationWFactor(builder, self.dilationWFactor)
        tflite.Conv3DOptionsAddDilationHFactor(builder, self.dilationHFactor)
        options = tflite.Conv3DOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.Conv3DOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class VarHandleOperator(BaseOperator):
    def __init__(self, inputs, outputs, container=(), sharedName=()) -> None:
        super().__init__(tflite.BuiltinOperator.VAR_HANDLE, inputs, outputs)
        self.container = container
        self.sharedName = sharedName

    def build(self, builder: flatbuffers.Builder) -> Offset:
        container = create_numpy_array(builder, tflite.VarHandleOptions.Container, self.container)
        sharedName = create_numpy_array(builder, tflite.VarHandleOptions.SharedName, self.sharedName)
        tflite.VarHandleOptionsStart(builder)
        tflite.VarHandleOptionsAddContainer(builder, container)
        tflite.VarHandleOptionsAddSharedName(builder, sharedName)
        options = tflite.VarHandleOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.VarHandleOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class ReadVariableOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.READ_VARIABLE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.ReadVariableOptionsStart(builder)
        options = tflite.ReadVariableOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.ReadVariableOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class AssignVariableOperator(BaseOperator):
    def __init__(
        self,
        inputs,
        outputs,
    ) -> None:
        super().__init__(tflite.BuiltinOperator.ASSIGN_VARIABLE, inputs, outputs)

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.AssignVariableOptionsStart(builder)
        options = tflite.AssignVariableOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.AssignVariableOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BroadcastArgsOperator(BaseOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(tflite.BuiltinOperator.BROADCAST_ARGS, inputs, outputs)


class RandomStandardNormalOperator(BaseOperator):
    def __init__(self, inputs, outputs, seed=0, seed2=0) -> None:
        super().__init__(tflite.BuiltinOperator.RANDOM_STANDARD_NORMAL, inputs, outputs)
        self.seed = seed
        self.seed2 = seed2

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RandomOptionsStart(builder)
        tflite.RandomOptionsAddSeed(builder, self.seed)
        tflite.RandomOptionsAddSeed2(builder, self.seed2)
        options = tflite.RandomOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RandomOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class BucketizeOperator(BaseOperator):
    def __init__(self, inputs, outputs, boundaries=()) -> None:
        super().__init__(tflite.BuiltinOperator.BUCKETIZE, inputs, outputs)
        self.boundaries = boundaries

    def build(self, builder: flatbuffers.Builder) -> Offset:
        boundaries = create_numpy_array(builder, tflite.BucketizeOptions.Boundaries, self.boundaries)
        tflite.BucketizeOptionsStart(builder)
        tflite.BucketizeOptionsAddBoundaries(builder, boundaries)
        options = tflite.BucketizeOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BucketizeOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class RandomUniformOperator(BaseOperator):
    def __init__(self, inputs, outputs, seed=0, seed2=0) -> None:
        super().__init__(tflite.BuiltinOperator.RANDOM_UNIFORM, inputs, outputs)
        self.seed = seed
        self.seed2 = seed2

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RandomOptionsStart(builder)
        tflite.RandomOptionsAddSeed(builder, self.seed)
        tflite.RandomOptionsAddSeed2(builder, self.seed2)
        options = tflite.RandomOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RandomOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class MultinomialOperator(BaseOperator):
    def __init__(self, inputs, outputs, seed=0, seed2=0) -> None:
        super().__init__(tflite.BuiltinOperator.MULTINOMIAL, inputs, outputs)
        self.seed = seed
        self.seed2 = seed2

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.RandomOptionsStart(builder)
        tflite.RandomOptionsAddSeed(builder, self.seed)
        tflite.RandomOptionsAddSeed2(builder, self.seed2)
        options = tflite.RandomOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.RandomOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op


class GeluOperator(BaseOperator):
    def __init__(self, inputs, outputs, approximate=False) -> None:
        super().__init__(tflite.BuiltinOperator.GELU, inputs, outputs)
        self.approximate = approximate

    def build(self, builder: flatbuffers.Builder) -> Offset:
        tflite.GeluOptionsStart(builder)
        tflite.GeluOptionsAddApproximate(builder, self.approximate)
        options = tflite.GeluOptionsEnd(builder)

        tfl_inputs_idx = create_numpy_array(builder, tflite.Operator.Inputs, self.tfl_inputs_idx)
        tfl_outputs_idx = create_numpy_array(builder, tflite.Operator.Outputs, self.tfl_outputs_idx)

        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, self.op.index)
        tflite.OperatorAddInputs(builder, tfl_inputs_idx)
        tflite.OperatorAddOutputs(builder, tfl_outputs_idx)
        tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.GeluOptions)
        tflite.OperatorAddBuiltinOptions(builder, options)
        self.tfl_op = tflite.OperatorEnd(builder)

        return self.tfl_op
