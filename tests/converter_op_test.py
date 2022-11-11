import sys
import unittest
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import IS_CI

from tinynn.converter import TFLiteConverter


def assert_close(actual, expected, *args, **kwargs):
    if hasattr(torch.testing, 'assert_close'):
        torch.testing.assert_close(actual, expected, *args, **kwargs)
    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('check_')}
        if 'msg' in filtered_kwargs:
            msg = filtered_kwargs['msg']
            if not isinstance(msg, str):
                filtered_kwargs['msg'] = msg()

        if not isinstance(actual, (tuple, list)):
            actual_v = [actual]
            expected_v = [expected]
        else:
            actual_v = actual
            expected_v = expected

        for a, e in zip(actual_v, expected_v):
            if not kwargs.get('check_dtype', True):
                t = torch.promote_types(a.dtype, e.dtype)
                a = a.to(dtype=t)
                e = e.to(dtype=t)
            torch.testing.assert_allclose(a, e, *args, **filtered_kwargs)


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
        tfl_outputs = []
        for i in range(len(outputs)):
            arr = np.asarray(interpreter.get_tensor(output_details[i]['index']))
            o = torch.from_numpy(arr)
            tfl_outputs.append(o)
        return tfl_outputs
    else:
        arr = np.asarray(interpreter.get_tensor(output_details[0]['index']))
        return torch.from_numpy(arr)


def get_model_path():
    size = getattr(get_model_path, 'size', 0)
    model_path = f'out/converter_op_{size}.tflite'
    setattr(get_model_path, 'size', size + 1)
    return model_path


def u8_to_s8(t):
    if t.dtype == torch.quint8:
        t = torch.int_repr(t)
    t_i32 = t.to(dtype=torch.int32)
    t_i32 -= 128
    return t_i32.to(dtype=torch.int8)


class ConverterOPTester(unittest.TestCase):
    def test_masked_fill(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.finfo(x.dtype).min)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.tensor(1.0))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor_with_cast(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, torch.tensor(1))
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_masked_fill_tensor_with_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.masked_fill(x, x > 0.5, x[0, 0, 0, 0])
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_binary_elementwise_same_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x, y):
                return func(x, y)

            model_path = get_model_path()
            inputs = [dummy_input, dummy_input_1]
            converter = TFLiteConverter(model, inputs, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(*inputs)
            tfl_output = tfl_run_model(model_path, inputs, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    def test_binary_elementwise_constant_same_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dummy_input_1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    def test_binary_elementwise_different_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randint(1, 10, size=dummy_input.shape)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x, y):
                return func(x, y)

            model_path = get_model_path()
            inputs = [dummy_input, dummy_input_1]
            converter = TFLiteConverter(model, inputs, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(*inputs)
            tfl_output = tfl_run_model(model_path, inputs, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    def test_binary_elementwise_constant_different_dtype(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randint(1, 10, size=dummy_input.shape)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dummy_input_1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    def test_binary_elementwise_scalar_int(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, 1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    def test_binary_elementwise_scalar_float(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'add'),
            (torch, 'mul'),
            (torch, 'sub'),
            (torch, 'div'),
            (torch, 'greater'),
            (torch, 'remainder'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
            (torch, 'rsub'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, 1.0)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, equal_nan=True)

    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion('1.7.0'), "torch.where cannot take scalar inputs")
    def test_where_int_scalars(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, 0, 1)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_dtype=False)

    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion('1.7.0')
        or LooseVersion(torch.__version__) >= LooseVersion('1.12.0'),
        "torch.where cannot take scalar inputs",
    )
    def test_where_float_scalars(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, -1.5, 0.5)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_dtype=False)

    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion('1.7.0'), "torch.where cannot take scalar inputs")
    def test_where_tensor_scalar(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, x.double(), 0.5)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_dtype=False)

    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion('1.7.0'), "torch.where cannot take scalar inputs")
    def test_where_scalar_tensor(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, 0.5, x.double())
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_where_tensors(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.where(x > 0.5, -x, x)
                return y

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_reduce_ops_no_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'mean'),
            (torch, 'sum'),
            (torch, 'min'),
            (torch, 'max'),
            (torch, 'prod'),
            (torch, 'amin'),
            (torch, 'amax'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_reduce_ops_single_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'mean'),
            (torch, 'sum'),
            (torch, 'min'),
            (torch, 'max'),
            (torch, 'prod'),
            (torch, 'amin'),
            (torch, 'amax'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                res = func(x, dim=1)
                return res if type(res) == torch.Tensor else res[0]

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg)

    def test_reduce_ops_single_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'mean'),
            (torch, 'sum'),
            (torch, 'min'),
            (torch, 'max'),
            (torch, 'prod'),
            (torch, 'amin'),
            (torch, 'amax'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                res = func(x, dim=1, keepdim=True)
                return res if type(res) == torch.Tensor else res[0]

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg)

    def test_reduce_ops_multi_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'mean'),
            (torch, 'sum'),
            (torch, 'amin'),
            (torch, 'amax'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=[1, 2])

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_reduce_ops_multi_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'mean'),
            (torch, 'sum'),
            (torch, 'amin'),
            (torch, 'amax'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=[1, 2], keepdim=True)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_unary_bitwise_ops(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _not(x):
            return ~x

        funcs = [torch.bitwise_not, _not]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_binary_bitwise_ops_scalar(self):
        raise unittest.SkipTest('Cannot go through torch.jit.trace')
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _and(x, y):
            return x & y

        def _or(x, y):
            return x | y

        funcs = [torch.bitwise_and, torch.bitwise_or, _and, _or]

        for func in funcs:
            print(f'testing {func.__name__}')
            for val in (False, True):

                def model(x):
                    return func(x, val)

                model_path = get_model_path()
                converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
                converter.convert()

                dummy_output = model(dummy_input)
                tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
                assert_close(dummy_output, tfl_output)

    def test_binary_bitwise_ops_tensor(self):
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0
        dummy_input_2 = torch.randn(1, 3, 224, 224, dtype=torch.float32) > 0

        def _and(x, y):
            return x & y

        def _or(x, y):
            return x | y

        funcs = [torch.bitwise_and, torch.bitwise_or, _and, _or]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x, y):
                return func(x, y)

            model_path = get_model_path()
            dummy_input = (dummy_input_1, dummy_input_2)
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(*dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_activation_ops_no_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (F, 'relu'),
            (F, 'relu6'),
            (F, 'hardsigmoid'),
            (F, 'hardswish'),
            (F, 'hardtanh'),
            (F, 'glu'),
            (F, 'silu'),
            (F, 'tanh'),
            (F, 'sigmoid'),
            (F, 'softplus'),
            (F, 'elu'),
            (F, 'leaky_relu'),
            (F, 'mish'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_activation_ops_approx_no_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.gelu]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            with self.assertRaisesRegex(AssertionError, r'.* (are not close!|exceeded the margin of error).*'):
                assert_close(dummy_output, tfl_output)

    def test_prelu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = nn.PReLU()

            def forward(self, x):
                return self.prelu(x)

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_prelu_multi_channel(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = nn.PReLU(3)

            def forward(self, x):
                return self.prelu(x)

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_prelu_multi_channel_with_transposes(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = nn.PReLU(16)

            def forward(self, x):
                x = F.interpolate(x, scale_factor=2)
                x = self.prelu(x)
                x = x.permute([0, 2, 3, 1])
                return x

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 16, 128, 1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_prelu_with_reshapes(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = nn.PReLU()

            def forward(self, x):
                x = x.reshape(1, 16, 128)
                x = self.prelu(x)
                x = x.reshape([1, 16, 128, 1])
                return x

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 16, 128, 1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_prelu_multi_channel_with_reshapes(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = nn.PReLU(16)

            def forward(self, x):
                x = x.reshape(1, 16, 128)
                x = self.prelu(x)
                x = x.reshape([1, 16, 128, 1])
                return x

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 16, 128, 1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_prelu_0d(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = torch.nn.PReLU(48)

            def forward(self, x):
                return self.prelu(x)

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 48)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_prelu_0d_with_reshapes(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = torch.nn.PReLU(48)

            def forward(self, x):
                x = x.reshape(1, 48, 1)
                x = self.prelu(x)
                x = x.reshape(1, 48)
                return x

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 48)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_softmax_like_funcs(self):
        dummy_input = torch.randn(1, 1000, dtype=torch.float32)

        funcs = [F.softmax, F.log_softmax]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=-1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_softmax_like_funcs_non_last_dim(self):
        dummy_input = torch.randn(1, 1000, 1, 1, dtype=torch.float32)

        funcs = [F.softmax, F.log_softmax]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_argminmax(self):
        dummy_input = torch.randperm(1 * 3 * 10 * 10).to(dtype=torch.float32).reshape(1, 3, 10, 10)

        funcs = [torch.argmin, torch.argmax]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_argminmax_keepdim(self):
        dummy_input = torch.randperm(1 * 3 * 10 * 10).to(dtype=torch.float32).reshape(1, 3, 10, 10)

        funcs = [torch.argmin, torch.argmax]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=1, keepdim=True)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, check_dtype=False)

    def test_noops(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [nn.Dropout().eval(), torch.Tensor.clone, torch.Tensor.contiguous]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.8.0'), 'multinomial is not supported')
    def test_dropout_with_training_mode(self):
        dummy_input = torch.randn(9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.2, True)

        model = Model()
        model.train()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.8.0'), 'multinomial is not supported')
    def test_dropout_3d_with_training_mode(self):
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.2, True)

        model = Model()
        model.train()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    def test_float_unary_ops(self):
        random_val = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        min_val = torch.tensor(0.1, dtype=torch.float32)
        dummy_input = torch.clamp(random_val, min=min_val)

        funcs = [torch.reciprocal, torch.exp, torch.log, torch.sqrt, torch.rsqrt, torch.sin, torch.cos, torch.floor]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_pow_scalar(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.pow(x, 2)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_pow_tensor(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        exponents = [torch.randint(0, 4, dummy_input.shape), torch.randint(0, 4, (1,))]

        for exponent in exponents:

            def model(x):
                return torch.pow(x, exponent)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_reshape_ops_no_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.flatten, torch.squeeze]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_flatten_with_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.flatten(x, 1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_reshape_ops_with_shapes(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.reshape, torch.Tensor.view]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, (1, -1, 1))

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_squeeze_with_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.squeeze(x, 0)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_unsqueeze_with_args(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.unsqueeze(x, -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_transpose(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.transpose(x, 0, -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_t_1d(self):
        dummy_input = torch.randn(4, dtype=torch.float32)

        def model(x):
            return torch.t(x)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_t_2d(self):
        dummy_input = torch.randn(4, 3, dtype=torch.float32)

        def model(x):
            return torch.t(x)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_permute(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return x.permute(0, 2, 3, 1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_clamp_hardtanh(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.clamp, F.hardtanh]
        ranges = [(0, 6), (0, 1), (-1.5, 1.5), (0, None), (None, 0), (-1.5, None), (None, 1.5)]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')
            for min_val, max_val in ranges:
                if func != torch.clamp and None in (min_val, max_val):
                    continue

                def model(x):
                    return func(x, min_val, max_val)

                model_path = get_model_path()
                converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
                converter.convert()

                dummy_output = model(dummy_input)
                tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
                assert_close(dummy_output, tfl_output)

    def test_clamp_minmax(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        func_names = [
            (torch, 'clamp_min'),
            (torch, 'clamp_max'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        funcs = [torch.clamp_min, torch.clamp_max]
        ranges = [0, -1.5, 1.5]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')
            for val in ranges:

                def model(x):
                    return func(x, val)

                model_path = get_model_path()
                converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
                converter.convert()

                dummy_output = model(dummy_input)
                tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
                assert_close(dummy_output, tfl_output)

    def test_flip_single_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.flip(x, [1])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_flip_multi_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.flip(x, [1, 2])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_cat_no_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.cat([x])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_cat_self_no_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.cat([x, x])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_cat_self_negative_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.cat([x, x], -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_cat_constant_negative_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.cat([x, dummy_input_1], -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_cat_tensors_negative_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        dummy_input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x, y):
            return torch.cat([x, y], -1)

        model_path = get_model_path()
        dummy_input = (dummy_input, dummy_input_1)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_stack_no_arg(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.stack([x])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_stack_self(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.stack([x, x])

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_stack_self_negative_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.stack([x, x], -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_floor_div_scalar(self):
        dummy_input = torch.randint(0, 100, size=(1, 3, 224, 224)).int()

        def model(x):
            return torch.floor_divide(x, 2)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_floor_div_tensor(self):
        dummy_input = torch.randint(0, 100, size=(1, 3, 224, 224)).int()
        dummy_input_1 = torch.tensor(2).int()

        def model(x):
            return torch.floor_divide(x, dummy_input_1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_floor_div_tensor_different_dtype(self):
        dummy_input = torch.randint(0, 100, size=(1, 3, 224, 224)).int()
        dummy_input_1 = torch.tensor(2)

        def model(x, y):
            return torch.floor_divide(x, y)

        model_path = get_model_path()
        dummy_input = (dummy_input, dummy_input_1)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'pixel_shuffle'), "Pixel shuffle is not supported")
    def test_pixel_shuffle_no_reorder(self):
        dummy_input = torch.randn(1, 9, 1, 1, dtype=torch.float32)

        def model(x):
            return torch.pixel_shuffle(x, 3)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'pixel_unshuffle'), "Pixel unshuffle is not supported")
    def test_pixel_unshuffle_no_reorder(self):
        dummy_input = torch.randn(1, 1, 3, 3, dtype=torch.float32)

        def model(x):
            return torch.pixel_unshuffle(x, 3)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    @unittest.skipIf(not hasattr(torch, 'pixel_shuffle'), "Pixel shuffle is not supported")
    def test_pixel_shuffle_with_reorder(self):
        dummy_input = torch.randn(1, 36, 7, 7, dtype=torch.float32)

        def model(x):
            return torch.pixel_shuffle(x, 3)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'pixel_unshuffle'), "Pixel unshuffle is not supported")
    def test_pixel_unshuffle_with_reorder(self):
        dummy_input = torch.randn(1, 12, 21, 21, dtype=torch.float32)

        def model(x):
            return torch.pixel_unshuffle(x, 3)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_embedding_1d(self):
        dummy_input = torch.randint(0, 100, size=(100,))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(100, 24)

            def forward(self, x):
                return self.emb(x)

        model = Model()
        model.eval()
        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_embedding_2d(self):
        dummy_input = torch.randint(0, 100, size=(10, 10))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(100, 24)

            def forward(self, x):
                return self.emb(x)

        model = Model()
        model.eval()
        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_embedding_3d(self):
        dummy_input = torch.randint(0, 1000, size=(10, 10, 10))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(1000, 24)

            def forward(self, x):
                return self.emb(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_embedding_3d_with_padding_idx(self):
        dummy_input = torch.randint(0, 1000, size=(10, 10, 10))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(1000, 24, padding_idx=0)

            def forward(self, x):
                return self.emb(x)

        model = Model()
        model.eval()
        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_split_chunk_divisible(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        funcs = [torch.split, torch.Tensor.chunk, torch.chunk]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 3, dim=1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, check_stride=False)

    def test_split_chunk_non_divisible(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        funcs = [torch.split, torch.Tensor.chunk, torch.chunk]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 5, dim=1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, check_stride=False)

    def test_split_chunk_negative_dim(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        funcs = [torch.split, torch.Tensor.chunk, torch.chunk]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 7, dim=-1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output, check_stride=False)

    def test_split_with_sizes(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        def model(x):
            return torch.split(x, [200, 24], dim=-1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_chunk_divisible(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        def model(x):
            return list(torch.chunk(x, 3, 1))

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_chunk_indivisible(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        def model(x):
            return list(torch.chunk(x, 7, 1))

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_chunk_indivisible_negative_dim(self):
        dummy_input = torch.randn(1, 9, 224, 224, dtype=torch.float32)

        def model(x):
            return list(torch.chunk(x, 11, -1))

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_repeat_single_dim(self):
        dummy_input = torch.randn(10, dtype=torch.float32)

        def model(x):
            return x.repeat(4)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_repeat_multi_dim(self):
        dummy_input = torch.randn(10, dtype=torch.float32)

        def model(x):
            return x.repeat(4, 2)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch.Tensor, 'expand_as'), 'expand_as is not available')
    def test_expand_as(self):
        dummy_input = torch.randn(3, 1, dtype=torch.float32)
        dummy_input_1 = torch.randn(3, 4, dtype=torch.float32)

        def model(x, y):
            return x.expand_as(y)

        inputs = [dummy_input, dummy_input_1]

        model_path = get_model_path()
        converter = TFLiteConverter(model, inputs, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*inputs)
        tfl_output = tfl_run_model(model_path, inputs, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_expand_simple(self):
        dummy_input = torch.randn(3, 1, dtype=torch.float32)

        def model(x):
            return x.expand(3, 4)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_expand_negative_dim(self):
        dummy_input = torch.randn(3, 1, dtype=torch.float32)

        def model(x):
            return x.expand(-1, 4)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_expand_more_dims(self):
        dummy_input = torch.randn(3, 1, dtype=torch.float32)

        def model(x):
            return x.expand(2, -1, 4)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_expand_noop(self):
        dummy_input = torch.randn(3, 1, dtype=torch.float32)

        def model(x):
            return x.expand(-1, -1)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_copy_constant(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        buffer = torch.empty(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return buffer.copy_(x)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_copy_constant_broadcast(self):
        dummy_input = torch.randn(224, dtype=torch.float32)
        buffer = torch.empty(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return buffer.copy_(x)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_copy_constant_broadcast_with_cast(self):
        dummy_input = torch.randint(0, 100, size=(224,), dtype=torch.int32)
        buffer = torch.empty(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return buffer.copy_(x)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_2d_matmuls_constant(self):
        dummy_input = torch.randn(9, 17, dtype=torch.float32)
        mat = torch.randn(17, 22, dtype=torch.float32)

        funcs = [torch.mm, torch.matmul]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, mat)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_2d_matmuls_tensor(self):
        mat1 = torch.randn(9, 17, dtype=torch.float32)
        mat2 = torch.randn(17, 22, dtype=torch.float32)

        funcs = [torch.mm, torch.matmul]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x, y):
                return func(x, y)

            dummy_input = (mat1, mat2)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(*dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_3d_matmuls_constant(self):
        dummy_input = torch.randn(1, 9, 17, dtype=torch.float32)
        mat = torch.randn(1, 17, 22, dtype=torch.float32)

        funcs = [torch.bmm, torch.matmul]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, mat)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_3d_matmuls_tensor(self):
        mat1 = torch.randn(1, 9, 17, dtype=torch.float32)
        mat2 = torch.randn(1, 17, 22, dtype=torch.float32)

        funcs = [torch.bmm, torch.matmul]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x, y):
                return func(x, y)

            dummy_input = (mat1, mat2)
            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(*dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_3d_2d_matmul_constant(self):
        dummy_input = torch.randn(1, 9, 17, dtype=torch.float32)
        mat = torch.randn(17, 22, dtype=torch.float32)

        def model(x):
            return torch.matmul(x, mat)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_3d_2d_matmul_tensor(self):
        mat1 = torch.randn(1, 9, 17, dtype=torch.float32)
        mat2 = torch.randn(17, 22, dtype=torch.float32)

        def model(x, y):
            return torch.matmul(x, y)

        dummy_input = (mat1, mat2)
        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_2d_linear(self):
        dummy_input = torch.randn(9, 17, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(17, 22)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_3d_linear(self):
        dummy_input = torch.randn(1, 9, 17, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(17, 22)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_2d_linear_no_bias(self):
        dummy_input = torch.randn(9, 17, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(17, 22, bias=False)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_3d_linear_no_bias(self):
        dummy_input = torch.randn(1, 9, 17, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(17, 22, bias=False)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_addmm(self):
        dummy_input = torch.randn(9, 17, dtype=torch.float32)
        mat = torch.randn(17, 22, dtype=torch.float32)
        bias = torch.randn(22, dtype=torch.float32)

        def model(x):
            return torch.addmm(bias, x, mat)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_pooling(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.avg_pool2d, F.max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 7)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_pooling_negative_values(self):
        dummy_input = -torch.abs(torch.randn(1, 3, 224, 224, dtype=torch.float32))

        funcs = [F.avg_pool2d, F.max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 11, 7, 2)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_pooling_same(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.avg_pool2d, F.max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 11, 7, 2)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_pooling_with_pad(self):
        dummy_input = torch.randn(1, 3, 220, 222, dtype=torch.float32)

        funcs = [F.avg_pool2d, F.max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 7, padding=(2, 1))

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_pooling_with_stride(self):
        dummy_input = torch.randn(1, 3, 220, 222, dtype=torch.float32)

        funcs = [F.avg_pool2d, F.max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 7, 7)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_max_pool_with_ceil_mode(self):
        dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)

        def model(x):
            return F.max_pool2d(x, 3, 2, 0, ceil_mode=False)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_avg_pool_with_count_include_pad(self):
        dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)

        def model(x):
            return F.avg_pool2d(x, 7, padding=(2, 1), count_include_pad=False)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_adaptive_pool(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.adaptive_avg_pool2d, F.adaptive_max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 32)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_adaptive_pool_2(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.adaptive_avg_pool2d, F.adaptive_max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, (16, 32))

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_adaptive_pool_3(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.adaptive_avg_pool2d, F.adaptive_max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, (None, 32))

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_adaptive_pool_4(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [F.adaptive_avg_pool2d, F.adaptive_max_pool2d]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x, 1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
            assert_close(dummy_output, tfl_output)

    def test_constant_pad_1d(self):
        dummy_input = torch.randn(1, 3, 224, dtype=torch.float32)

        def model(x):
            return F.pad(x, (1, 1))

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_constant_pad_2d(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.pad(x, (1, 1, 2, 2))

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_constant_pad_2d_with_fill_value(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.pad(x, (1, 1, 2, 2), 'constant', 0.5)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_reflection_pad_1d(self):
        dummy_input = torch.randn(1, 3, 224, dtype=torch.float32)

        def model(x):
            return F.pad(x, (2, 2), 'reflect')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_reflection_pad_2d(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.pad(x, (1, 1, 2, 0), 'reflect')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_upsample_bilinear(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, scale_factor=2.0, mode='bilinear')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(
        LooseVersion(torch.__version__) >= '1.13.0' and LooseVersion(torch.__version__) < '1.13.1',
        "See https://github.com/pytorch/pytorch/issues/87968",
    )
    def test_upsample_bilinear_align_corners(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_upsample_nearest(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, scale_factor=2.0, mode='nearest')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_upsample_bilinear_output_size(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, size=(448, 448), mode='bilinear')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_upsample_nearest_output_size(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, size=(448, 448), mode='nearest')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_upsample_nearest_output_size_complex(self):
        dummy_input = torch.randn(1, 3, 8, 22, dtype=torch.float32)

        def model(x):
            return F.interpolate(x, size=(15, 43), mode='nearest')

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_lstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_lstm_no_bias(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, bias=False)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_lstm_batch_first(self):
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, batch_first=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_lstm_multi_layer(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_lstm_multi_layer_no_bias(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2, bias=False)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm_no_bias(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, bias=False, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm_batch_first(self):
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, batch_first=True, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_bilstm_multi_layer(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm_multi_layer_no_bias(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2, bidirectional=True, bias=False)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm_multi_layer_as_lstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, map_bilstm_to_lstm=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_bilstm_multi_layer_no_bias_as_lstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 2, bidirectional=True, bias=False)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, map_bilstm_to_lstm=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_sigmoid_(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x):
            x.sigmoid_()
            return x

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input.clone())
        tfl_output = tfl_run_model(model_path, dummy_input.clone(), dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_type_as(self):
        dummy_input_1 = torch.randint(1, 100, (1, 3, 224, 224), dtype=torch.int32)
        dummy_input_2 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x, y):
            return x.type_as(y)

        model_path = get_model_path()
        dummy_input = (dummy_input_1, dummy_input_2)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_to(self):
        dummy_input_1 = torch.randint(1, 100, size=(1, 3, 224, 224), dtype=torch.int32)
        dummy_input_2 = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        def model(x, y):
            return x.to(y.dtype)

        model_path = get_model_path()
        dummy_input = (dummy_input_1, dummy_input_2)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[2:4]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice_no_end(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[2:]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice_no_start(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[:4]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice_negative_start(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[-1:]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice_negative_end(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[:-1]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_slice_with_step(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[1:-1:2]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False)

    def test_select(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[0]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_select_negative_index(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[-1]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_index(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[[2, 3]]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_index_negative_index(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[[-1, -2]]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_index_tensor_indices(self):
        dummy_input_1 = torch.randn(10, 10, dtype=torch.float32)
        dummy_input_2 = torch.randint(0, 10, size=(10,))

        def model(x, y):
            return x[y]

        model_path = get_model_path()

        dummy_input = (dummy_input_1, dummy_input_2)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_index_multi_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return x[..., [2, 3]]

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_gather(self):
        dummy_input = torch.randn(10, dtype=torch.float32)

        def model(x):
            return torch.gather(x, 0, torch.tensor([1, 2, 3]))

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_gather_negative_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.gather(x, -1, torch.tensor([[1, 2, 3]]))

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_gather_tensor_indices(self):
        dummy_input_1 = torch.randn(10, 10, dtype=torch.float32)
        dummy_input_2 = torch.randint(0, 10, size=(10, 2))

        def model(x, y):
            return torch.gather(x, 0, y)

        model_path = get_model_path()

        dummy_input = (dummy_input_1, dummy_input_2)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(*dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.1.0'), 'scatter_nd is not supported')
    def test_scatter(self):
        dummy_input = torch.randn(10, dtype=torch.float32)

        def model(x):
            return torch.zeros(10, dtype=torch.float32).scatter_(0, torch.tensor([1, 2, 3]), x)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.1.0'), 'scatter_nd is not supported')
    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion('1.7.0')
        or LooseVersion(torch.__version__) >= LooseVersion('1.12.0'),
        "torch.Tensor.scatter_ cannot take scalar inputs",
    )
    def test_scatter_scalar(self):
        dummy_input = torch.randperm(10)[:3]

        def model(x):
            return torch.zeros(10, dtype=torch.float32).scatter_(0, x, 1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.1.0'), 'scatter_nd is not supported')
    def test_scatter_negative_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.zeros(10, 10, dtype=torch.float32).scatter_(-1, torch.tensor([[1, 2, 3]]), x)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_var_std_no_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            func_name = func.__name__ if hasattr(func, '__name__') else type(func).__name__
            print(f'testing {func_name}')

            def model(x):
                return func(x)

            model_path = get_model_path()

            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_var_std_ops_single_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=1)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg)

    def test_var_std_ops_single_dim_unbiased(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=-3, unbiased=False)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg)

    def test_var_std_single_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=1, keepdim=True)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg)

    def test_var_std_multi_dim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=[1, 2])

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_var_std_multi_dim_keepdim(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [torch.std, torch.var]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, dim=[1, 2], keepdim=True)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {func.__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_norms(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        funcs = [nn.BatchNorm2d(3), nn.BatchNorm2d(3, affine=False), nn.InstanceNorm2d(3), nn.LayerNorm([3, 224, 224])]

        for func in funcs:

            class Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.norm = func

                def forward(self, x):
                    return self.norm(x)

            model = Model()
            model.eval()

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

            def msg(*args, **kwargs):
                return f'testing {type(func).__name__} failed: {args}'

            assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(3, 8, 3)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_same_pad(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(3, 3, 11, 7, 2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_same_pad_complex(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(3, 3, (8, 12), (3, 5), (3, 4))

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_no_bias(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(3, 8, 3, bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_group_conv(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(4, 8, 3, groups=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_group_conv_no_bias(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(4, 8, 3, groups=2, bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_group_deconv(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(4, 8, 3, groups=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_group_deconv_no_bias(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(4, 8, 3, groups=2, bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_depthwise_conv(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(4, 4, 3, groups=4)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_depthwise_conv_no_bias(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(4, 4, 3, groups=4, bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_depthwise_conv_same_pad(self):
        dummy_input = torch.randn(1, 4, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv2d(4, 4, 11, 7, 2, groups=4)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_transpose(self):
        dummy_input = torch.randn(1, 16, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_transpose_same_pad(self):
        dummy_input = torch.randn(1, 16, 32, 32, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(16, 33, 11, stride=7, padding=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_transpose_no_bias(self):
        dummy_input = torch.randn(1, 16, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_transpose_with_group(self):
        # dummy_input = torch.zeros(1, 4, 50, 100, dtype=torch.float32)
        dummy_input = ((torch.arange(1 * 4 * 50 * 100) % 255 - 128) / 128.0).float().reshape(1, 4, 50, 100)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose2d(4, 4, 1, groups=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    def test_conv_pixelshuffle(self):
        dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 1),
                    nn.PixelShuffle(2),
                )

            def forward(self, x):
                return self.block(x)

        model_path = get_model_path()

        model = Model()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    def test_topk(self):
        def model(x):
            return x.topk(5)

        model_path = get_model_path()
        dummy_input = torch.randn((2, 10), dtype=torch.float32)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(tfl_output, dummy_output, check_dtype=False)

    def test_im2col(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.unfold = nn.Unfold(3, 2, 1, 2)

            def forward(self, x):
                return self.unfold(x)

        model_path = get_model_path()

        model = Model()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.1.0'), 'scatter_nd is not supported')
    def test_col2im(self):
        dummy_input = torch.randn(1, 12, 12, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fold = nn.Fold((4, 5), (2, 2), (1, 1), (0, 0), (1, 1))

            def forward(self, x):
                return self.fold(x)

        model_path = get_model_path()

        model = Model()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.4.0'), 'cumsum is not supported')
    def test_cumsum(self):
        def model(x):
            return torch.cumsum(x, 1)

        model_path = get_model_path()
        dummy_input = torch.randn((2, 10), dtype=torch.float32)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(tfl_output, dummy_output, check_dtype=False)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.4.0'), 'cumsum is not supported')
    def test_cumsum_negative_dim(self):
        def model(x):
            return torch.cumsum(x, -1)

        model_path = get_model_path()
        dummy_input = torch.randn((2, 10), dtype=torch.float32)
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(tfl_output, dummy_output, check_dtype=False)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.5.0'), 'conv3d is not supported')
    def test_conv3d(self):
        dummy_input = torch.randn(1, 16, 10, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=0)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.6.0'), 'pad with 5d-dim is not supported')
    def test_conv3d_with_pad(self):
        dummy_input = torch.randn(1, 16, 10, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.6.0'), 'conv3d is not supported')
    def test_conv3d_no_bias(self):
        dummy_input = torch.randn(1, 16, 10, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0), bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.5.0'), 'conv3d is not supported')
    def test_conv3d_same_pad(self):
        dummy_input = torch.randn(1, 3, 14, 14, 14, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Conv3d(3, 3, 11, stride=7, padding=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.6.0'), 'conv_transpose3d is not supported')
    def test_conv_transpose3d(self):
        dummy_input = torch.randn(1, 16, 10, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.6.0'), 'conv_transpose3d is not supported')
    def test_conv_transpose3d_no_bias(self):
        dummy_input = torch.randn(1, 16, 10, 50, 100, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2), bias=False)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.6.0'), 'conv_transpose3d is not supported')
    def test_conv_transpose3d_same_pad(self):
        dummy_input = torch.randn(1, 3, 2, 2, 2, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.ConvTranspose3d(3, 3, 11, stride=7, padding=2)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)

        def msg(*args, **kwargs):
            return f'testing failed: {args}'

        assert_close(dummy_output, tfl_output, msg=msg, atol=256.0, rtol=256.0, equal_nan=True)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, 1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_1d(self):
        dummy_input = torch.randn(100, dtype=torch.float32)

        def model(x):
            return torch.roll(x, 1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_negative_shift(self):
        dummy_input = torch.arange(10 * 10).to(dtype=torch.float32).reshape(10, 10)

        def model(x):
            return torch.roll(x, -1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_noop(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, 100)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_noop_1(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, -100)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_single_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, 1, 0)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_negative_shift_single_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, -1, 0)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_negative_shift_single_negative_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, -1, -1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'roll'), "Roll is not supported")
    def test_roll_multi_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.roll(x, [-1, 1], [0, 1])

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'round'), "Round is not supported")
    def test_round(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.round(x)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p1(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=1)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p1_with_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=1, dim=0)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p1_with_dim_keepdim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=1, dim=0, keepdim=True)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p2(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=2)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input).view(1)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p2_with_dim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=2, dim=0)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch, 'norm'), "Norm is not supported")
    def test_norm_p2_with_dim_keepdim(self):
        dummy_input = torch.randn(10, 10, dtype=torch.float32)

        def model(x):
            return torch.norm(x, p=2, dim=0, keepdim=True)

        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)


class ConverterQuantizedOPTester(unittest.TestCase):
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

    def test_quantize(self):
        def model(x):
            return torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1.0, rtol=1.0)

    def test_quantize_int8(self):
        def model(x):
            return torch.quantize_per_tensor(x, 0.5, 128, torch.quint8)

        dummy_input = torch.randn(1, 3, 224, 224)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(torch.int_repr(model(dummy_input)))
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1.0, rtol=1.0)

    def test_dequantize(self):
        def model(x):
            return torch.dequantize(x)

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = model(dummy_input)
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=0.5, rtol=0.5)

    def test_dequantize_int8(self):
        def model(x):
            return torch.dequantize(x)

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=0.5, rtol=0.5)

    def test_quantized_add(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_add_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_add_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add_relu(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_add_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add_relu(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_add_scalar(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add_scalar(x, 0.5)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_add_scalar_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.add_scalar(x, 0.5)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_mul(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.mul(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_mul_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.mul(x, x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_mul_scalar(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.mul_scalar(x, 0.5)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_mul_scalar_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.mul_scalar(x, 0.5)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_cat(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()

            def forward(self, x):
                return self.q_func.cat([x, x], -1)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    def test_quantized_cat_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.QFunctional()
                # TFLite requires same qparams for all inputs and outputs
                self.q_func.scale = 0.5
                self.q_func.zero_point = 128

            def forward(self, x):
                return self.q_func.cat([x, x], 1)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Hardswish'), 'Quantized hardswish is not supported')
    def test_quantized_hardswish(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Hardswish(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Hardswish'), 'Quantized hardswish is not supported')
    def test_quantized_hardswish_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Hardswish(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ELU'), 'Quantized elu is not supported')
    def test_quantized_elu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ELU(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ELU'), 'Quantized elu is not supported')
    @unittest.skipIf(LooseVersion(tf.__version__) < LooseVersion('2.4.1'), 'Quantized elu is not supported')
    def test_quantized_elu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ELU(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ReLU6'), 'Quantized relu6 is not supported')
    def test_quantized_relu6(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ReLU6()

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ReLU6'), 'Quantized relu6 is not supported')
    def test_quantized_relu6_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ReLU6()

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'LeakyReLU'), 'Quantized leaky_relu is not supported')
    def test_quantized_leaky_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.LeakyReLU(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'LeakyReLU'), 'Quantized leaky_relu is not supported')
    def test_quantized_leaky_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.LeakyReLU(0.5, 128)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=1, rtol=1)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_depthwise_conv2d_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1, groups=3, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_depthwise_conv2d_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1, groups=3, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_depthwise_conv1d_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 3, 1, groups=3, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_depthwise_conv1d_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 3, 1, groups=3, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_depthwise_conv2d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_depthwise_conv2d_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_depthwise_conv1d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_depthwise_conv1d_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU2d'), 'Quantized conv2d_relu is not supported')
    def test_quantized_depthwise_conv2d_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU2d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU2d'), 'Quantized conv2d_relu is not supported')
    def test_quantized_depthwise_conv2d_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU2d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU1d'), 'Quantized conv1d_relu is not supported')
    def test_quantized_depthwise_conv1d_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU1d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU1d'), 'Quantized conv1d_relu is not supported')
    def test_quantized_depthwise_conv1d_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU1d(3, 3, 1, groups=3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_per_channel_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1)
                q_w = self.q_func.weight()
                q_b = self.q_func.bias()
                w = q_w.dequantize()
                new_w = torch.quantize_per_channel(
                    w, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0, 0, 0]), 0, torch.qint8
                )
                self.q_func.set_weight_bias(new_w, q_b)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_depthwise_conv2d_per_channel_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 3, 1, groups=3)
                q_w = self.q_func.weight()
                q_b = self.q_func.bias()
                w = q_w.dequantize()
                new_w = torch.quantize_per_channel(
                    w, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0, 0, 0]), 0, torch.qint8
                )
                self.q_func.set_weight_bias(new_w, q_b)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 1, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 1, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_conv1d_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 1, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_conv1d_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 1, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_with_group(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(4, 4, 1, groups=2)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv2d_with_group_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(4, 4, 1, groups=2)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(
            model, dummy_input, model_path, nchw_transpose=False, group_conv_rewrite=True, quantize_target_type='int8'
        )
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_conv1d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv1d'), 'Quantized conv1d is not supported')
    def test_quantized_conv1d_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv1d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU2d'), 'Quantized conv2d_relu is not supported')
    def test_quantized_conv2d_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU2d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU2d'), 'Quantized conv2d_relu is not supported')
    def test_quantized_conv2d_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU2d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv_pixelshuffle_per_channel_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(4, 4, 1)
                q_w = self.q_func.weight()
                q_b = self.q_func.bias()
                w = q_w.dequantize()
                new_w = torch.quantize_per_channel(
                    w, torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([0, 0, 0, 0]), 0, torch.qint8
                )
                self.q_func.set_weight_bias(new_w, q_b)
                self.q_pixs = torch.nn.PixelShuffle(2)

            def forward(self, x):
                return self.q_pixs(self.q_func(x))

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 48, 48), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv_pixelshuffle(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(4, 4, 1)
                self.q_pixs = torch.nn.PixelShuffle(2)

            def forward(self, x):
                return self.q_pixs(self.q_func(x))

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 48, 48), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Conv2d'), 'Quantized conv2d is not supported')
    def test_quantized_conv_pixelshuffle_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Conv2d(4, 4, 1)
                self.q_pixs = torch.nn.PixelShuffle(2)

            def forward(self, x):
                return self.q_pixs(self.q_func(x))

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 48, 48), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU2d'), 'Quantized conv2d_relu is not supported')
    def test_quantized_conv_relu_pixelshuffle(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU2d(4, 4, 1)
                self.q_pixs = torch.nn.PixelShuffle(2)

            def forward(self, x):
                return self.q_pixs(self.q_func(x))

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 4, 48, 48), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU1d'), 'Quantized conv1d_relu is not supported')
    def test_quantized_conv1d_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU1d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'ConvReLU1d'), 'Quantized conv1d_relu is not supported')
    def test_quantized_conv1d_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.ConvReLU1d(3, 1, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_quantized_linear(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Linear(10, 10)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_quantized_linear_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Linear(10, 10)
                print(self.q_func)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_quantized_linear_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Linear(10, 10, bias_=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'Linear'), 'Quantized linear is not supported')
    def test_quantized_linear_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.Linear(10, 10, bias_=False)
                print(self.q_func)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'LinearReLU'), 'Quantized linear_relu is not supported')
    def test_quantized_linear_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.LinearReLU(10, 10)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'LinearReLU'), 'Quantized linear_relu is not supported')
    def test_quantized_linear_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.LinearReLU(10, 10)
                print(self.q_func)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'LinearReLU'), 'Quantized linear_relu is not supported')
    def test_quantized_linear_relu_no_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.LinearReLU(10, 10, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'LinearReLU'), 'Quantized linear_relu is not supported')
    def test_quantized_linear_relu_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.LinearReLU(10, 10, bias=False)
                print(self.q_func)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 10), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose2d'), 'Quantized conv_transpose2d is not supported')
    def test_quantized_conv_transpose2d_no_bias(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose2d(3, 3, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose2d'), 'Quantized conv_transpose2d is not supported')
    def test_quantized_conv_transpose2d_no_bias_int8(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose2d(3, 3, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose1d'), 'Quantized conv_transpose1d is not supported')
    def test_quantized_conv_transpose1d_no_bias(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose1d(3, 3, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose1d'), 'Quantized conv_transpose1d is not supported')
    def test_quantized_conv_transpose1d_no_bias_int8(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose1d(3, 3, 1, bias=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose2d'), 'Quantized conv_transpose2d is not supported')
    def test_quantized_conv_transpose2d(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose2d(3, 3, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose2d'), 'Quantized conv_transpose2d is not supported')
    def test_quantized_conv_transpose2d_int8(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose2d(3, 3, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose1d'), 'Quantized conv_transpose1d is not supported')
    def test_quantized_conv_transpose1d(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose1d(3, 3, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'ConvTranspose1d'), 'Quantized conv_transpose1d is not supported')
    def test_quantized_conv_transpose1d_int8(self):
        if self.backend == 'fbgemm':
            raise unittest.SkipTest('Quantized conv_transpose is unsupported in FBGEMM')

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.ConvTranspose1d(3, 3, 1)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'BatchNorm2d'), 'Quantized batch_norm2d is not supported')
    def test_quantized_batch_norm2d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.BatchNorm2d(3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized, 'BatchNorm2d'), 'Quantized batch_norm2d is not supported')
    def test_quantized_batch_norm2d_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.BatchNorm2d(3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'BNReLU2d'), 'Quantized bn_relu2d is not supported')
    def test_quantized_batch_norm2d_relu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.BNReLU2d(3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
        converter.convert()

        dummy_output = torch.int_repr(model(dummy_input))
        dummy_input = torch.int_repr(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.intrinsic.quantized, 'BNReLU2d'), 'Quantized bn_relu2d is not supported')
    def test_quantized_batch_norm2d_relu_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.intrinsic.quantized.BNReLU2d(3)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.quantize_per_tensor(torch.randn(1, 3, 224, 224), 0.5, 128, torch.quint8)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = u8_to_s8(model(dummy_input))
        dummy_input = u8_to_s8(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256, rtol=256, check_stride=False)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'Linear'), 'Quantized linear_dynamic is not supported')
    def test_quantized_linear_dynamic_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.dynamic.Linear(10, 10)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'Linear'), 'Quantized linear_dynamic is not supported')
    def test_quantized_linear_dynamic_no_bias_int8(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_func = torch.nn.quantized.dynamic.Linear(10, 10, bias_=False)

            def forward(self, x):
                return self.q_func(x)

        model = Model()
        model.eval()

        dummy_input = torch.randn(1, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_lstm_dynamic(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(10, 20)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_lstm_dynamic_batch_first(self):
        raise unittest.SkipTest('TFLite hybrid LSTM kernel with batch_first=True is broken')
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, batch_first=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_lstm_dynamic_multi_layer(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, 2)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic_batch_first(self):
        raise unittest.SkipTest('TFLite hybrid LSTM kernel with batch_first=True is broken')
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, batch_first=True, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic_multi_layer(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, 2, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8')
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic_as_lstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(
            model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8', map_bilstm_to_lstm=True
        )
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic_batch_first_as_lstm(self):
        raise unittest.SkipTest('TFLite hybrid LSTM kernel with batch_first=True is broken')
        dummy_input = torch.randn(1, 9, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, batch_first=True, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(
            model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8', map_bilstm_to_lstm=True
        )
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, check_stride=False, atol=256.0, rtol=256.0)

    @unittest.skipIf(not hasattr(torch.nn.quantized.dynamic, 'LSTM'), 'Quantized lstm is not supported')
    def test_bilstm_dynamic_multi_layer_as_lstm(self):
        dummy_input = torch.randn(9, 1, 10, dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.quantized.dynamic.LSTM(10, 20, 2, bidirectional=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = Model()
        model.eval()

        model_path = get_model_path()
        converter = TFLiteConverter(
            model, dummy_input, model_path, nchw_transpose=False, quantize_target_type='int8', map_bilstm_to_lstm=True
        )
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output, atol=256.0, rtol=256.0)

    def test_comparisons_with_scalar(self):
        dummy_input = torch.quantize_per_tensor(
            torch.randn(1, 3, 224, 224, dtype=torch.float32), 0.5, 128, torch.quint8
        )

        func_names = [
            (torch, 'greater'),
            (torch, 'less'),
            (torch, 'greater_equal'),
            (torch, 'less_equal'),
            (torch, 'eq'),
            (torch, 'ne'),
        ]

        funcs = [getattr(ns, attr) for ns, attr in func_names if hasattr(ns, attr)]

        for func in funcs:
            print(f'testing {func.__name__}')

            def model(x):
                return func(x, 0)

            model_path = get_model_path()
            converter = TFLiteConverter(model, dummy_input, model_path, nchw_transpose=False)
            converter.convert()

            dummy_output = model(dummy_input)
            tfl_input = torch.int_repr(dummy_input)
            tfl_output = tfl_run_model(model_path, tfl_input, dummy_output)
            assert_close(dummy_output, tfl_output)


if __name__ == '__main__':
    unittest.main()
