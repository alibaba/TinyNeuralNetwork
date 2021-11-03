import torch
import numpy as np

from .quantized_schema import *
from .. import CommonGraph
from .. import tflite as tfl

import tflite as tfl_schema


class QuantizedRelu6Operator(QuantizedRelu6Schema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.Relu6Operator, graph_converter)


class QuantizedMulScalarOperator(QuantizedMulScalarSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedMulOperator.parse_common(self, node, attrs, args, graph_converter)


class QuantizedMulOperator(QuantizedMulSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)

    def parse_common(self, node, attrs, args, graph_converter):
        other = self.input_tensors[1]
        if type(other) not in (int, float):
            self.elementwise_binary(tfl.MulOperator, graph_converter)
        elif other in (1.0, 1):
            self.passthrough(graph_converter)
        else:
            assert type(other) in (int, float)
            other_tensor = torch.tensor([other], dtype=torch.float)
            self.input_names[1] = self.get_unique_attr_name()
            if not other_tensor.is_nonzero():
                self.input_tensors[1] = torch.quantize_per_tensor(other_tensor, 0.5, 128, torch.quint8)
            elif (torch.sign(other_tensor) < 0).all():
                self.input_tensors[1] = torch.quantize_per_tensor(
                    other_tensor, -other_tensor[0] / 127, 255, torch.quint8)
            else:
                self.input_tensors[1] = torch.quantize_per_tensor(other_tensor, other_tensor[0] / 127, 0, torch.quint8)
            self.elementwise_binary(tfl.MulOperator, graph_converter)


class QuantizedConv2dReluOperator(QuantizedConv2dReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedConv2dOperator.parse_common(self, graph_converter, tfl_schema.ActivationFunctionType.RELU)


class QuantizedConv2dOperator(QuantizedConv2dSchema):
    def parse_common(self, graph_converter, fusedActivation=tfl_schema.ActivationFunctionType.NONE, transpose=False):
        input_tensor = self.find_or_create_input(0, graph_converter)
        params, _ = self.unpack_params(self.input_tensors[1])
        weight, bias = params['weight'], params['bias']

        weight_dim = weight.dim()
        output_padding = [0] * (weight_dim - 2)

        if len(self.input_tensors) > 4:
            stride, padding, dilation, groups = self.input_tensors[2:6]
        else:
            stride = params['stride']
            padding = params['padding']
            dilation = params['dilation']
            groups = params['groups']
            output_padding = params.get('output_padding', output_padding)

        weight_tensor = self.create_attr_tensor(weight)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        output_tensor = outputs[0]

        self.rescale_weight_scale_for_qnnpack(input_tensor, weight_tensor, output_tensor)

        # Bias handling
        bias_scale = input_tensor.quantization.scale * weight_tensor.quantization.scale
        if transpose:
            bias = self.quantize(bias, bias_scale, 0, dtype=torch.uint8)
        else:
            bias = self.quantize(bias, bias_scale, 0, dtype=torch.int32)

        bias_tensor = self.create_attr_tensor(bias)

        inputs = [input_tensor, weight_tensor, bias_tensor]

        if transpose:
            assert fusedActivation == tfl_schema.ActivationFunctionType.NONE
            graph_converter.add_operator(tfl.GenericTransposeConvOperator(inputs, outputs, stride, padding,
                                                                          dilation, output_padding, groups))
        else:
            graph_converter.add_operator(tfl.GenericConvOperator(inputs, outputs, stride, padding,
                                                                 dilation, output_padding, groups, fusedActivation))

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(graph_converter)


class QuantizedConv1dReluOperator(QuantizedConv1dReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedConv2dOperator.parse_common(self, graph_converter, tfl_schema.ActivationFunctionType.RELU)


class QuantizedCatOperator(QuantizedCatSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)

        dim = self.input_tensors[1]
        assert type(dim) == int

        if dim < 0:
            dim += self.input_tensors[0].ndim

        names = graph_converter.get_list_expanded_names(self.input_names[0])
        inputs = self.to_tfl_tensors(names, self.input_tensors[0], graph_converter=graph_converter)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        graph_converter.add_operator(tfl.ConcatenationOperator(inputs, outputs, dim))


class QuantizedBatchNorm1dOperator(QuantizedBatchNorm1dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        eps = self.input_tensors[args['eps']]
        inputs = [self.find_or_create_input(i, graph_converter) for i in range(5)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = self.wrap_ops_with_dequant_quants([tfl.BatchNormOperator(inputs, outputs, eps)])
        for op in ops:
            graph_converter.add_operator(op)


class QuantizedBatchNorm2dOperator(QuantizedBatchNorm2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        eps = self.input_tensors[args['eps']]
        inputs = [self.find_or_create_input(i, graph_converter) for i in range(5)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = self.wrap_ops_with_dequant_quants([tfl.BatchNormOperator(inputs, outputs, eps)])
        for op in ops:
            graph_converter.add_operator(op)


class QuantizedBatchNorm2dReluOperator(QuantizedBatchNorm2dReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        eps = self.input_tensors[args['eps']]
        inputs = [self.find_or_create_input(i, graph_converter) for i in range(5)]
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)

        ops = self.wrap_ops_with_dequant_quants([tfl.BatchNormOperator(
            inputs, outputs, eps, tfl_schema.ActivationFunctionType.RELU)])
        for op in ops:
            graph_converter.add_operator(op)


class QuantizedAddScalarOperator(QuantizedAddScalarSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedAddOperator.parse_common(self, node, attrs, args, graph_converter)


class QuantizedConv1dOperator(QuantizedConv1dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedConv2dOperator.parse_common(self, graph_converter)


class QuantizedLinearOperator(QuantizedLinearSchema):
    def parse_common(self, graph_converter, fusedActivation=tfl_schema.ActivationFunctionType.NONE):
        _, state = self.unpack_params(self.input_tensors[1])
        input_tensor = self.find_or_create_input(0, graph_converter)
        weight = state[0][0]
        bias = state[0][1]

        weight_tensor = self.create_attr_tensor(weight)
        outputs = self.to_tfl_tensors(self.output_names, self.output_tensors)
        output_tensor = outputs[0]

        self.rescale_weight_scale_for_qnnpack(input_tensor, weight_tensor, output_tensor)

        # Bias handling
        bias_scale = input_tensor.quantization.scale * weight_tensor.quantization.scale
        bias = self.quantize(bias, bias_scale, 0, dtype=torch.int32)
        bias_tensor = self.create_attr_tensor(bias)

        inputs = [input_tensor, weight_tensor, bias_tensor]

        graph_converter.add_operator(tfl.FullyConnectedOperator(
            inputs, outputs, fusedActivationFunction=fusedActivation))

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(graph_converter)


class QuantizedAddOperator(QuantizedAddSchema):
    def parse_common(self, node, attrs, args, graph_converter):
        other = self.input_tensors[1]
        if type(other) not in (int, float):
            self.elementwise_binary(tfl.AddOperator, graph_converter)
        elif other in (0.0, 0):
            self.passthrough(graph_converter)
        else:
            other_tensor = torch.tensor([other], dtype=torch.float)
            self.input_names[1] = self.get_unique_attr_name()
            if (torch.sign(other_tensor) < 0).all():
                self.input_tensors[1] = torch.quantize_per_tensor(
                    other_tensor, -other_tensor[0] / 127, 255, torch.quint8)
            else:
                self.input_tensors[1] = torch.quantize_per_tensor(other_tensor, other_tensor[0] / 127, 0, torch.quint8)
            self.elementwise_binary(tfl.AddOperator, graph_converter)

    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.parse_common(node, attrs, args, graph_converter)


class QuantizedLinearReluOperator(QuantizedLinearReluSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedLinearOperator.parse_common(self, graph_converter, tfl_schema.ActivationFunctionType.RELU)


class QuantizedConvTranspose2dOperator(QuantizedConvTranspose2dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedConv2dOperator.parse_common(self, graph_converter, transpose=True)


class QuantizedConvTranspose1dOperator(QuantizedConvTranspose1dSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        QuantizedConv2dOperator.parse_common(self, graph_converter, transpose=True)


class QuantizedHardswishOperator(QuantizedHardswishSchema):
    def parse(self, node, attrs, args, graph_converter):
        super().parse(node, attrs, args, graph_converter)

        self.run(node)
        self.elementwise_unary(tfl.HardSwishOperator, graph_converter)
