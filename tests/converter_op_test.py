import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import numpy as np

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
            arr = np.asarray(interpreter.get_tensor(output_details[i]['index']))
            o = torch.from_numpy(arr)
            outputs.append(o)
    else:
        arr = np.asarray(interpreter.get_tensor(output_details[0]['index']))
        return torch.from_numpy(arr)


def get_model_path():
    size = getattr(get_model_path, 'size', 0)
    model_path = f'out/converter_op_{size}.tflite'
    setattr(get_model_path, 'size', size + 1)
    return model_path


class ConverterOPTester(unittest.TestCase):
    def test_glu(self):
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

    def test_binary_elementwise_same_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x, y): return func(x, y)
            model_path = get_model_path()
            inputs = [dummy_input, dummy_input_1]
            converter = TFLiteConverter(model, inputs, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(*inputs)
            tfl_output = tfl_run_model(model_path, inputs, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_constant_same_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, dummy_input_1)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_different_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randint(1, 10, size=dummy_input.shape)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x, y): return func(x, y)
            model_path = get_model_path()
            inputs = [dummy_input, dummy_input_1]
            converter = TFLiteConverter(model, inputs, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(*inputs)
            tfl_output = tfl_run_model(model_path, inputs, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_constant_different_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randint(1, 10, size=dummy_input.shape)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, dummy_input_1)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_scalar_int(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, 1)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_scalar_float(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.add, torch.mul, torch.sub, torch.div, torch.greater, torch.remainder,
                 torch.less, torch.greater_equal, torch.less_equal, torch.eq, torch.ne]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, 1.0)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_where_int_scalars(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, 0, 1)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_where_float_scalars(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, -1.5, 0.5)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_where_tensor_scalar(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, x.double(), 0.5)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_where_scalar_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, 0.5, x.double())
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        torch.testing.assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_where_tensors(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, -x, x)
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

    def test_reduce_ops_no_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.mean, torch.sum, torch.min, torch.max, torch.prod, torch.amin, torch.amax]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            def msg(*args, **kwargs): return f'testing {func.__name__} failed: {args}'
            torch.testing.assert_close(dummy_output, tfl_output, msg=msg, atol=1e-4, rtol=1e-4)

    def test_reduce_ops_single_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.mean, torch.sum, torch.min, torch.max, torch.prod, torch.amin, torch.amax]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): 
                res = func(x, dim=1)
                return res if type(res) == torch.Tensor else res[0]
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            def msg(*args, **kwargs): return f'testing {func.__name__} failed: {args}'
            torch.testing.assert_close(dummy_output, tfl_output, msg=msg)

    def test_reduce_ops_single_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.mean, torch.sum, torch.min, torch.max, torch.prod, torch.amin, torch.amax]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): 
                res = func(x, dim=1)
                return res if type(res) == torch.Tensor else res[0]
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            def msg(*args, **kwargs): return f'testing {func.__name__} failed: {args}'
            torch.testing.assert_close(dummy_output, tfl_output, msg=msg)


    def test_reduce_ops_multi_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.mean, torch.sum, torch.amin, torch.amax]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, dim=[1,2])
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            def msg(*args, **kwargs): return f'testing {func.__name__} failed: {args}'
            torch.testing.assert_close(dummy_output, tfl_output, msg=msg, atol=1e-4, rtol=1e-4)

    def test_reduce_ops_multi_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.mean, torch.sum, torch.amin, torch.amax]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x, dim=[1,2], keepdim=True)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            def msg(*args, **kwargs): return f'testing {func.__name__} failed: {args}'
            torch.testing.assert_close(dummy_output, tfl_output, msg=msg, atol=1e-4, rtol=1e-4)


    def test_unary_bitwise_ops(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _not(x): return ~x

        funcs = [torch.bitwise_not, _not]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x): return func(x)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_bitwise_ops_scalar(self):
        raise unittest.SkipTest('Cannot go through torch.jit.trace')
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _and(x, y): return x & y
        def _or(x, y): return x | y

        funcs = [torch.bitwise_and, torch.bitwise_or, _and, _or]

        for func in funcs:
            print(f'testing {func.__name__}')
            for val in (False, True):
                def model(x): return func(x, val)
                model_path = get_model_path()
                converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
                converter.convert()

                dummy_output = model(dummy_input)
                tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
                torch.testing.assert_close(dummy_output, tfl_output)

    def test_binary_bitwise_ops_tensor(self):
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0
        dummy_input_2 = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _and(x, y): return x & y
        def _or(x, y): return x | y

        funcs = [torch.bitwise_and, torch.bitwise_or, _and, _or]

        for func in funcs:
            print(f'testing {func.__name__}')
            def model(x, y): return func(x, y)
            model_path = get_model_path()
            dummy_input = (dummy_input_1, dummy_input_2)
            converter = TFLiteConverter(model, dummy_input, model_path, input_transpose=False)
            converter.convert()

            dummy_output = model(*dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            torch.testing.assert_close(dummy_output, tfl_output)

if __name__ == '__main__':
    unittest.main()
