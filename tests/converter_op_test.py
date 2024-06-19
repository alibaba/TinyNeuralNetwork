import unittest

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter

# import tensorflow as t


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
    def test_missing_outputs_as_constants(self):
        class TestModel(nn.Module):
            def forward(self, x):
                y = x.relu()
                return y, torch.zeros_like(x)

        model = TestModel()
        model.eval()

        dummy_input = torch.randn(2, 10)
        model_path = get_model_path()

        converter = TFLiteConverter(model, dummy_input, model_path, missing_outputs_as_constants=True)
        converter.convert()

        dummy_output = model(dummy_input)
        tfl_output = tfl_run_model(model_path, dummy_input, dummy_output)
        assert_close(dummy_output, tfl_output)


if __name__ == '__main__':
    unittest.main()
