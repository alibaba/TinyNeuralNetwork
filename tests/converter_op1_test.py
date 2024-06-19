import unittest
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

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
    @unittest.skipIf(
        LooseVersion(tf.__version__) < '2.9.0', 'TFLite hybrid LSTM kernel with batch_first=True is broken'
    )
    def test_lstm_dynamic_batch_first(self):
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
    @unittest.skipIf(
        LooseVersion(tf.__version__) < '2.9.0', 'TFLite hybrid LSTM kernel with batch_first=True is broken'
    )
    def test_bilstm_dynamic_batch_first(self):
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
    @unittest.skipIf(
        LooseVersion(tf.__version__) < '2.9.0', 'TFLite hybrid LSTM kernel with batch_first=True is broken'
    )
    def test_bilstm_dynamic_batch_first_as_lstm(self):
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
