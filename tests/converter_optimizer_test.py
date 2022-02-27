import unittest

import tflite
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinynn.converter import TFLiteConverter


def parse_model(path):
    with open(path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    return model


def get_model_path():
    size = getattr(get_model_path, 'size', 0)
    model_path = f'out/converter_optimizer_{size}.tflite'
    setattr(get_model_path, 'size', size + 1)
    return model_path


class ConverterOptimizerTester(unittest.TestCase):
    def test_tuple_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).BuiltinCode(), (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT)
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

    def test_repeated_list_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                return list(y) + list(y)

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).BuiltinCode(), (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT)
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 6)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

    def test_input_output_with_noop(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.view(x.shape)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_branch_output_with_noop(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                return [t.view(t.shape) for t in y]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).BuiltinCode(), (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT)
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

    def test_branch_output_with_noop_complex(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                left = [t.view(t.shape) for t in y]
                right = [F.relu(t) for t in y]
                return list(y) + left + right

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        # TODO: Optimize this case

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 10)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 9)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 10)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

        split_output_indices = tfl_model.Subgraphs(0).Operators(0).OutputsAsNumpy().tolist()
        split_output_names = [tfl_model.Subgraphs(0).Tensors(i).Name() for i in split_output_indices]

        for i in range(1, 10):
            input_idx = tfl_model.Subgraphs(0).Operators(i).Inputs(0)
            input_name = tfl_model.Subgraphs(0).Tensors(input_idx).Name()
            self.assertIn(input_name, split_output_names)

    def test_simple_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = y.permute(0, 3, 1, 2)
                y = F.relu(y)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = F.relu(y)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.add(y, y)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.cat([y, y, y], -1)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.CONCATENATION)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_transpose_with_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = F.relu(y)
                z = y.permute(0, 3, 1, 2)
                z = F.relu(z)
                return y, z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(1).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(2).BuiltinCode(), tflite.BuiltinOperator.TRANSPOSE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        for i in range(3):
            self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_simple_reshape(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 224))
                y = torch.reshape(y, (1, 3, 224, 224))
                y = F.relu(y)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_reshape(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 224))
                y = F.relu(y)
                y = torch.reshape(y, (1, 3, 224, 224))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_reshape(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 224))
                y = torch.add(y, y)
                y = torch.reshape(y, (1, 3, 224, 224))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_reshape(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 224))
                y = torch.cat([y, y, y], -1)
                y = torch.reshape(y, (1, 3, 224, 224 * 3))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.CONCATENATION)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_reshape_with_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 224))
                y = F.relu(y)
                z = torch.reshape(y, (1, 3, 224, 224))
                z = F.relu(z)
                return y, z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(1).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(2).BuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        for i in range(3):
            self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_pad_with_paired_reshape_and_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.reshape(y, (224, 224, 3))
                y = F.pad(y, (1, 1), "constant", 0)
                y = torch.reshape(y, (1, 224, 224, 5))
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.PAD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fold_buffer(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.register_parameter('weight', nn.Parameter(torch.randn(50, 40, dtype=torch.float32)))
                self.register_parameter('bias', nn.Parameter(torch.randn(40, dtype=torch.float32)))

            def forward(self, x):
                y = torch.addmm(self.bias, x, self.weight)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 50)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fold_shared_buffer(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.register_parameter('weight', nn.Parameter(torch.randn(50, 40, dtype=torch.float32)))
                self.register_parameter('bias', nn.Parameter(torch.randn(40, dtype=torch.float32)))

            def forward(self, x):
                y = torch.cat([torch.addmm(self.bias, x, self.weight) for _ in range(5)], dim=0)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 50)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)

        for i in range(5):
            self.assertEqual(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).BuiltinCode(),
                tflite.BuiltinOperator.FULLY_CONNECTED,
            )
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(5).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.CONCATENATION,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(5).OutputsLength(), 1)

    def test_fuse_activation(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = F.relu(x + 1)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 50)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

        builtin_opts = tfl_model.Subgraphs(0).Operators(0).BuiltinOptions()
        self.assertIsNotNone(builtin_opts)

        opts = tflite.FullyConnectedOptions()
        opts.Init(builtin_opts.Bytes, builtin_opts.Pos)
        self.assertEqual(opts.FusedActivationFunction(), tflite.ActivationFunctionType.RELU)

    def test_fuse_matmul_add(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.register_parameter('weight', nn.Parameter(torch.randn(50, 40, dtype=torch.float32)))
                self.register_parameter('bias', nn.Parameter(torch.randn(40, dtype=torch.float32)))

            def forward(self, x):
                y = torch.matmul(x, self.weight)
                y = torch.add(y, self.bias)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 50)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_mm_add(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.register_parameter('weight', nn.Parameter(torch.randn(50, 40, dtype=torch.float32)))
                self.register_parameter('bias', nn.Parameter(torch.randn(40, dtype=torch.float32)))

            def forward(self, x):
                y = torch.mm(x, self.weight)
                y = torch.add(y, self.bias)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 50)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_transpose_across_channel_shuffle(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, [1, 2, 112, 224, 3])
                x = torch.transpose(x, 1, 2)
                x = torch.reshape(x, [1, 224, 224, 3])
                x = x.permute(0, 3, 1, 2)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )

    def test_transpose_across_channel_shuffle_with_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, [1, 2, 112, 224, 3])
                x = torch.transpose(x, 1, 2)
                y = torch.reshape(x, [1, 224, 224, 3])
                y = y.permute(0, 3, 1, 2)
                return x, y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 4)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(3).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(3).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(3).OpcodeIndex()).BuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )

    def test_transpose_across_squeeze(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, [224, 224, 3])
                x = x.permute(2, 0, 1)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_transpose_across_unsqueeze(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(1, 2, 0)
                x = torch.reshape(x, [1, 224, 224, 3])
                x = x.permute(0, 3, 1, 2)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_quant_dequant(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = F.relu(x)
                x = torch.dequantize(x)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False, fuse_quant_dequant=True)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_branch_expand_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y1 = y.permute(0, 3, 1, 2)
                y2 = y.permute(0, 3, 1, 2)
                y = y1 + y2
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).BuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_branch_expand_transpose_buffer(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('buf1', nn.Parameter(torch.randn(1, 3, 224, 224)))

            def forward(self, x):
                y = self.buf1.permute(0, 2, 3, 1)
                y1 = y.permute(0, 3, 1, 2)
                y2 = y.permute(0, 3, 1, 2)
                y = x + y1 + y2
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 2)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 2)
        for i in range(2):
            self.assertEqual(tfl_model.OperatorCodes(i).BuiltinCode(), tflite.BuiltinOperator.ADD)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).InputsLength(), 2)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).OutputsLength(), 1)

    def test_unused_tensors(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.split(x, 1, 1)
                return x[1:]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

    def test_unused_tensors_branch_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.split(x, 1, -1)
                return [t.permute([0, 3, 1, 2]) for t in x[1:]]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

    def test_unused_tensors_branch_reshape(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.reshape(x, [6, 224, 224])
                x = torch.split(x, 2, 0)
                return [t.reshape([1, 2, 224, 224]) for t in x[1:]]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 6, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()


if __name__ == '__main__':
    unittest.main()
