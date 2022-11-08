import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import IS_CI

from tinynn.converter import TFLiteConverter
from tinynn.converter.schemas.tflite import schema_generated as tflite
from tinynn.converter.utils.tflite import parse_model


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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

    def test_input_output_transpose(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.relu()
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_input_output_transpose_quantize(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.relu()
                x = x.dequantize()
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.RELU),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RELU,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_input_output_transpose_branch(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.split(x, 1, 1)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT, tflite.BuiltinOperator.SPLIT_V),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

    def test_input_output_transpose_branch_complex(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                z = torch.relu(y[0])
                return list(y) + [z]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        # TODO: Optimize this case

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

        self.assertIn(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT, tflite.BuiltinOperator.SPLIT_V),
        )

    def test_input_output_transpose_branch_quantize(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = torch.split(x, 1, 1)
                x = [t.dequantize() for t in x]
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 5)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (
                tflite.BuiltinOperator.DEQUANTIZE,
                tflite.BuiltinOperator.QUANTIZE,
                tflite.BuiltinOperator.SPLIT_V,
                tflite.BuiltinOperator.SPLIT,
            ),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 5)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertIn(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT),
        )
        for i in range(2, 5):
            self.assertEqual(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).DeprecatedBuiltinCode(),
                tflite.BuiltinOperator.DEQUANTIZE,
            )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_input_output_transpose_branch_quantize_complex(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                y = torch.split(x, 1, 1)
                y = [t.dequantize() for t in y]
                z = torch.relu(y[0])
                return y + [z]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

        split_output_indices = tfl_model.Subgraphs(0).Operators(1).OutputsAsNumpy().tolist()
        split_output_names = [tfl_model.Subgraphs(0).Tensors(i).Name() for i in split_output_indices]

        for i in range(2, 6):
            op_index = tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()
            op_code = tfl_model.OperatorCodes(op_index)

            if op_code == tflite.BuiltinOperator.DEQUANTIZE:
                input_idx = tfl_model.Subgraphs(0).Operators(i).Inputs(0)
                input_name = tfl_model.Subgraphs(0).Tensors(input_idx).Name()
                self.assertIn(input_name, split_output_names)

    def test_repeated_list_output(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.split(x, 1, 1)
                return list(y) + list(y)

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT),
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.SPLIT_V, tflite.BuiltinOperator.SPLIT),
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_dequantize_quantize(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.relu()
                x = x.dequantize()
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.RELU),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RELU,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantize_rewrite_bmm(self):
        class TestModel(nn.Module):
            def forward(self, x, y):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                y = torch.quantize_per_tensor(y, 0.5, 128, torch.quint8)
                y = y.dequantize()
                z = torch.matmul(x, y)
                z = torch.quantize_per_tensor(z, 0.5, 128, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = [torch.randn(2, 3, 224), torch.randn(2, 224, 3)]
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, rewrite_quantizable=True, quantize_target_type='int8'
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 4)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.BATCH_MATMUL),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 4)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.BATCH_MATMUL,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(3).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantize_rewrite_softmax(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                z = F.softmax(x, -1)
                z = torch.quantize_per_tensor(z, 1.0 / 256, 0, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, rewrite_quantizable=True, quantize_target_type='int8'
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.SOFTMAX),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.SOFTMAX,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantize_rewrite_log_softmax(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                z = F.log_softmax(x, -1)
                z = torch.quantize_per_tensor(z, 16.0 / 256, 255, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, rewrite_quantizable=True, quantize_target_type='int8'
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.LOG_SOFTMAX),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.LOG_SOFTMAX,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantize_rewrite_abs(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                z = x.abs()
                z = torch.quantize_per_tensor(z, 0.25, 0, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, rewrite_quantizable=True, quantize_target_type='int8'
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.ABS),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.ABS,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantize_rewrite_sum(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = x.dequantize()
                z = x.sum()
                z = torch.quantize_per_tensor(z, 0.5, 128, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, rewrite_quantizable=True, quantize_target_type='int8'
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (tflite.BuiltinOperator.DEQUANTIZE, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.SUM),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.SUM,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_transpose_gather(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = y[:, :, :, [2, 1, 0]]
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.GATHER)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_transpose_mean(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = y.mean(axis=1, keepdim=True)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.MEAN)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_transpose_tile(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                y = x.expand((-1, 20, 20, -1))
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 1, 1)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TILE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_transpose_as_unary(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter('param1', nn.Parameter(torch.randn(1)))

            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.add(y, self.param1)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_transpose_as_unary_swap(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter('param1', nn.Parameter(torch.randn(1)))

            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.add(self.param1, y)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_transpose_broadcast(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter('param1', nn.Parameter(torch.randn(3)))

            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.add(y, self.param1)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_binary_elementwise_transpose_broadcast_swap(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter('param1', nn.Parameter(torch.randn(3)))

            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                y = torch.add(self.param1, y)
                y = y.permute(0, 3, 1, 2)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.CONCATENATION)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_transpose_pack(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 1)
                y = torch.stack([y, y, y], -1)
                y = y.permute(0, 2, 1, 3)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(16, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.PACK)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_transpose_unpack(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 1)
                y = torch.unbind(y, 0)
                return [t.permute(1, 0) for t in y]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(16, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.UNPACK)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 16)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 16)

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TRANSPOSE)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_reshape_tile(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 1, 20))
                y = y.expand((-1, 10, -1))
                y = torch.reshape(y, (1, 3, 10, 20))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 1, 20)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TILE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_reshape_gather(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 20, 20))
                y = y[[2, 1, 0]]
                y = torch.reshape(y, (1, 3, 20, 20))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 20, 20)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.GATHER)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_unary_elementwise_reshape_mean(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 20, 20))
                y = y.mean(axis=1, keepdim=True)
                y = torch.reshape(y, (1, 3, 1, 20))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 20, 20)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.MEAN)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.CONCATENATION)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_reshape_pack(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224))
                y = torch.stack([y, y, y], -1)
                y = torch.reshape(y, (1, 3, 224, 3))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.PACK)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_multiple_elementwise_reshape_unpack(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.reshape(x, (3, 224, 3))
                y = torch.unbind(y, -1)
                return [t.reshape(1, 3, 224) for t in y]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 3)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.UNPACK)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.PAD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)

        for i in range(5):
            self.assertEqual(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).DeprecatedBuiltinCode(),
                tflite.BuiltinOperator.FULLY_CONNECTED,
            )
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(5).OpcodeIndex()).DeprecatedBuiltinCode(),
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.FULLY_CONNECTED)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_transpose_across_kernel_reorder(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, [1, 4, 4, 3])
                x = x.permute(0, 3, 1, 2)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 8, 2)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )

    def test_transpose_across_kernel_reorder_downward(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = torch.reshape(x, [1, 4, 4, 3, 1])
                x = x.permute(0, 3, 4, 1, 2)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 8, 2)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )

    def test_transpose_across_kernel_reorder_upward(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = x.permute(0, 3, 4, 1, 2)
                x = torch.reshape(x, [1, 4, 4, 3])
                x = x.permute(0, 3, 1, 2)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 1, 8, 2)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
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
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
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
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RESHAPE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(1).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(2).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.TRANSPOSE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(3).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(3).OutputsLength(), 1)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(3).OpcodeIndex()).DeprecatedBuiltinCode(),
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, fuse_quant_dequant=True)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_consective_quant_dequant(self):
        class TestModel(nn.Module):
            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = x.relu()
                x = torch.dequantize(x)
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = torch.dequantize(x)
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = F.relu(x)
                x = torch.dequantize(x)
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, fuse_quant_dequant=True)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 2)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RELU)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_quant_dequant_with_indices(self):
        class TestModel(nn.Module):
            def forward(self, x, y):
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = F.relu(x)
                x = torch.dequantize(x)
                y = torch.quantize_per_tensor(y, 0.05, 128, torch.quint8)
                y = F.relu(y)
                y = torch.dequantize(y)
                return x, y

        model = TestModel()
        model.eval()

        dummy_input = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
        model_path = get_model_path()

        converter = TFLiteConverter(
            model,
            dummy_input,
            model_path,
            nchw_transpose=False,
            fuse_quant_dequant=True,
            fuse_input_indices=[0],
            fuse_output_indices=[0],
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 4)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 4)
        for i in range(4):
            self.assertIn(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).DeprecatedBuiltinCode(),
                (tflite.BuiltinOperator.RELU, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.DEQUANTIZE),
            )
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).InputsLength(), 1)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).OutputsLength(), 1)

    def test_fuse_quant_dequant_with_type(self):
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

        converter = TFLiteConverter(
            model,
            dummy_input,
            model_path,
            nchw_transpose=False,
            fuse_quant_dequant=True,
            quantize_input_output_type='uint8',
            quantize_target_type='int8',
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.RELU,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_quant_dequant_with_type_and_indices(self):
        class TestModel(nn.Module):
            def forward(self, x, y):
                x = torch.quantize_per_tensor(x, 0.05, 128, torch.quint8)
                x = F.relu(x)
                x = torch.dequantize(x)
                y = torch.quantize_per_tensor(y, 0.05, 128, torch.quint8)
                y = F.relu(y)
                y = torch.dequantize(y)
                return x, y

        model = TestModel()
        model.eval()

        dummy_input = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
        model_path = get_model_path()

        converter = TFLiteConverter(
            model,
            dummy_input,
            model_path,
            nchw_transpose=False,
            fuse_quant_dequant=True,
            quantize_input_output_type='uint8',
            quantize_target_type='int8',
            fuse_input_indices=[0],
            fuse_output_indices=[0],
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)
        for i in range(4):
            self.assertIn(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).DeprecatedBuiltinCode(),
                (tflite.BuiltinOperator.RELU, tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.DEQUANTIZE),
            )
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).InputsLength(), 1)
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).OutputsLength(), 1)

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 2)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 2)
        for i in range(2):
            self.assertEqual(tfl_model.OperatorCodes(i).DeprecatedBuiltinCode(), tflite.BuiltinOperator.ADD)
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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.SPLIT_V)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Inputs(0)).Buffer(), 0)
        for i in range(3):
            self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Outputs(i)).Buffer(), 0)

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.SPLIT_V)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Inputs(0)).Buffer(), 0)
        for i in range(3):
            self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Outputs(i)).Buffer(), 0)

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

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.SPLIT_V)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Inputs(0)).Buffer(), 0)
        for i in range(3):
            self.assertEqual(tfl_model.Subgraphs(0).Tensors(tfl_model.Subgraphs(0).Operators(0).Outputs(i)).Buffer(), 0)

    def test_fuse_simple_slice(self):
        class TestModel(nn.Module):
            def forward(self, x):
                return x[:, :, 1:-1, 1:-1]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.SLICE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_strided_slice(self):
        class TestModel(nn.Module):
            def forward(self, x):
                return x[:, :, 1:-1:2, 1:-1:2]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.STRIDED_SLICE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_multiple_strided_slice(self):
        class TestModel(nn.Module):
            def forward(self, x):
                return x[:, :, 1:-1:3, ::4][:, :, ::4, 1:-1:3]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.STRIDED_SLICE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_mixed_slice_and_strided_slice(self):
        class TestModel(nn.Module):
            def forward(self, x):
                return x[:, :, 1:-1, :][:, :, ::4, 1:-1:3]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.STRIDED_SLICE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_fuse_mixed_slice_and_strided_slice_reverse(self):
        class TestModel(nn.Module):
            def forward(self, x):
                return x[:, :, ::4, 1:-1:3][:, :, 1:-1, :]

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 1)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.STRIDED_SLICE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).InputsLength(), 4)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_cat_split_rewrite(self):
        class TestModel(nn.Module):
            def forward(self, x):
                v = torch.split(x, 1, -1)
                return torch.cat(v, -1)

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 32)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, tflite_micro_rewrite=True)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (
                tflite.BuiltinOperator.SPLIT_V,
                tflite.BuiltinOperator.SPLIT,
            ),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.SPLIT_V,
        )
        for i in range(1, 6):
            self.assertEqual(
                tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(i).OpcodeIndex()).DeprecatedBuiltinCode(),
                tflite.BuiltinOperator.CONCATENATION,
            )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 32)
        for i in range(1, 4):
            self.assertEqual(tfl_model.Subgraphs(0).Operators(i).InputsLength(), 10)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(4).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(5).InputsLength(), 4)


class ConverterOptimizerQuantizedTester(unittest.TestCase):
    backend: str

    def setUp(self):
        backends = ['qnnpack', 'fbgemm']
        for backend in backends:
            if IS_CI and backend == 'qnnpack':
                continue
            if backend in torch.backends.quantized.supported_engines:
                self.backend = backend
                torch.backends.quantized.engine = backend
                return
        self.skipTest('No quantization backend is found')

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_requantize_fuse(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.quantized.Linear(10, 10)

            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                x = self.linear(x)
                x = x.dequantize()
                x = torch.quantize_per_tensor(x, 0.3, 128, torch.quint8)
                x = x.dequantize()
                return x

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (
                tflite.BuiltinOperator.DEQUANTIZE,
                tflite.BuiltinOperator.QUANTIZE,
                tflite.BuiltinOperator.FULLY_CONNECTED,
            ),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.FULLY_CONNECTED,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_requantize_no_fuse(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.quantized.Linear(10, 10)
                self.functional = torch.nn.quantized.QFunctional()

            def forward(self, x):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                y = self.linear(x)
                x = y.dequantize()
                x = torch.quantize_per_tensor(x, 0.3, 128, torch.quint8)
                z = self.functional.add(x, y)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, tflite_micro_rewrite=True)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 6)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.FULLY_CONNECTED,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(3).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(4).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.ADD,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(5).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_quantizable_rewrite(self):
        class TestModel(nn.Module):
            def forward(self, x, y):
                x = torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)
                y = torch.quantize_per_tensor(y, 0.5, 128, torch.quint8)
                x = x.dequantize()
                y = y.dequantize()
                z = torch.matmul(x, y)
                z = torch.quantize_per_tensor(z, 0.5, 128, torch.quint8)
                z = z.dequantize()
                return z

        model = TestModel()
        model.eval()

        dummy_input = (torch.randn(1, 3, 10), torch.randn(1, 10, 3))
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, quantize_target_type='int8', rewrite_quantizable=True
        )
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 4)
        self.assertIn(
            tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(),
            (
                tflite.BuiltinOperator.DEQUANTIZE,
                tflite.BuiltinOperator.QUANTIZE,
                tflite.BuiltinOperator.BATCH_MATMUL,
            ),
        )
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 2)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 4)
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(0).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(1).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.QUANTIZE,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(2).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.BATCH_MATMUL,
        )
        self.assertEqual(
            tfl_model.OperatorCodes(tfl_model.Subgraphs(0).Operators(3).OpcodeIndex()).DeprecatedBuiltinCode(),
            tflite.BuiltinOperator.DEQUANTIZE,
        )
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_conv2d_gather(self):
        class TestModel(nn.Module):
            def __init__(self, with_bias=False):
                super(TestModel, self).__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 1, bias=with_bias),
                    nn.PixelShuffle(2),
                    nn.Conv2d(4, 4, 3, 1, 1, bias=with_bias),
                    nn.PixelShuffle(2),
                )

            def forward(self, x):
                return self.block(x)

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 128, 128)
        model_path = get_model_path()

        nchw_transpose = False
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=nchw_transpose)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 6)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.CONV_2D)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.DEPTH_TO_SPACE)
        self.assertEqual(tfl_model.OperatorCodes(3).DeprecatedBuiltinCode(), tflite.BuiltinOperator.CONV_2D)
        self.assertEqual(tfl_model.OperatorCodes(4).DeprecatedBuiltinCode(), tflite.BuiltinOperator.DEPTH_TO_SPACE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)

    def test_lower_transpose_dim_pass(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(0, 1, 2, 4, 3)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(2, 10, 10, 10, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, max_transpose_dims=4)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TRANSPOSE)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_lower_transpose_dim_pass_1(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(1, 0, 2, 4, 3)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 10, 10, 10, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, max_transpose_dims=4)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TRANSPOSE)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)

    def test_lower_transpose_dim_pass_2(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.permute(1, 0, 2, 4, 3)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(10, 1, 10, 10, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, max_transpose_dims=4)
        converter.convert()

        tfl_model = parse_model(model_path)
        self.assertEqual(tfl_model.OperatorCodesLength(), 3)
        self.assertEqual(tfl_model.OperatorCodes(0).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.OperatorCodes(1).DeprecatedBuiltinCode(), tflite.BuiltinOperator.TRANSPOSE)
        self.assertEqual(tfl_model.OperatorCodes(2).DeprecatedBuiltinCode(), tflite.BuiltinOperator.RESHAPE)
        self.assertEqual(tfl_model.SubgraphsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).InputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OutputsLength(), 1)
        self.assertEqual(tfl_model.Subgraphs(0).OperatorsLength(), 3)
        self.assertEqual(tfl_model.Subgraphs(0).Operators(0).OutputsLength(), 1)


if __name__ == '__main__':
    unittest.main()
