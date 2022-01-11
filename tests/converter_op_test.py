import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

from tinynn.converter import TFLiteConverter


def tfl_run_model(path, inputs, outputs):
    interpreter = tf.lite.Interpreter(model_path=path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if isinstance(inputs, (list, tuple)):
        for i, t in enumerate(inputs):
            interpreter.set_tensor(input_details[i]['index'], t.numpy())
    else:
        interpreter.set_tensor(input_details[0]['index'], inputs.numpy())

    interpreter.invoke()

    if isinstance(outputs, (list, tuple)):
        outputs = []
        for i in range(len(outputs)):
            o = torch.from_numpy(interpreter.get_tensor(output_details[i]['index']))
            outputs.append(o)
    else:
        return torch.from_numpy(interpreter.get_tensor(output_details[0]['index']))


def get_model_path():
    size = getattr(get_model_path, 'size', 0)
    model_path = f'out/converter_op_{size}.tflite'
    setattr(get_model_path, 'size', size + 1)
    return model_path


class ConverterOPTester(unittest.TestCase):
    def test_gelu(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.nn.functional.glu(x, -1)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output)

    def test_masked_fill(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.finfo(x.dtype).min)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.tensor(1.0))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor_with_cast(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.tensor(1))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor_with_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, x[0, 0, 0, 0])
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output)


if __name__ == '__main__':
    unittest.main()
