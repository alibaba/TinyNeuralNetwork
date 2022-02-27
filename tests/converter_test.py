import torch
import torchvision
import torchvision.models

import gc
import io
import inspect
import logging
import os
import unittest
import warnings

import numpy as np

from tinynn.converter import TFLiteConverter
from common_utils import collect_custom_models, collect_torchvision_models, prepare_inputs


HAS_TF = False
try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    pass


def data_to_tf(inputs, input_transpose):
    tf_inputs = list(map(lambda x: x.cpu().detach().numpy(), inputs))
    for i in range(len(tf_inputs)):
        if input_transpose[i]:
            tf_inputs[i] = np.transpose(tf_inputs[i], [0, 2, 3, 1])
    return tf_inputs


def get_tflite_out(model_path, inputs):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[i]['index'], inputs[i])

    interpreter.invoke()

    outputs = []
    for i in range(len(output_details)):
        output_data = interpreter.get_tensor(output_details[i]['index'])
        if not isinstance(output_data, np.ndarray):
            output_data = np.asarray(output_data).reshape((1,))
        outputs.append(output_data)

    return outputs


# Unsupported models
# resnext: group convs
# regnet: group convs
# yolov4: 5-d slices
BLACKLIST = (
    'resnext50_32x4d',
    'resnext101_32x8d',
    'Build_Model',
    'regnet_x_16gf',
    'regnet_x_1_6gf',
    'regnet_x_32gf',
    'regnet_x_3_2gf',
    'regnet_x_400mf',
    'regnet_x_800mf',
    'regnet_x_8gf',
    'regnet_y_16gf',
    'regnet_y_1_6gf',
    'regnet_y_32gf',
    'regnet_y_3_2gf',
    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_8gf',
)


class TestModelMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_classes = collect_torchvision_models()
        for test_class in test_classes:
            test_name = f'test_torchvision_model_{test_class.__name__}'
            simple_test_name = test_name + '_simple'
            d[simple_test_name] = mcls.build_model_test(test_class)
        test_classes = collect_custom_models()
        for test_class in test_classes:
            test_name = f'test_custom_model_{test_class.__name__}'
            simple_test_name = test_name + '_simple'
            d[simple_test_name] = mcls.build_model_test(test_class)
        return d

    @classmethod
    def build_model_test(cls, model_class):
        def f(self):
            model_name = model_class.__name__
            model_file = model_name
            model_file += '_simple'

            if model_name in BLACKLIST:
                raise unittest.SkipTest('IN BLACKLIST')

            if os.path.exists(f'out/{model_file}.tflite'):
                raise unittest.SkipTest('TESTED')

            m = model_class()
            m.eval()

            inputs = prepare_inputs(m)

            with torch.no_grad():
                out_path = f'out/{model_file}.tflite'
                out_pt = f'out/{model_file}.pt'
                converter = TFLiteConverter(m, inputs, out_path, dump_jit_model_path=out_pt, gc_when_reload=True)

                # Remove original model to lower memory usage
                del m

                converter.convert()

                os.remove(out_pt)

                if HAS_TF:
                    outputs = converter.get_outputs()
                    input_transpose = converter.input_transpose
                    input_tf = data_to_tf(inputs, input_transpose)
                    tf_outputs = get_tflite_out(out_path, input_tf)
                    output_tensors = list(map(torch.from_numpy, tf_outputs))
                    for pt, tt in zip(outputs, output_tensors):
                        result = torch.allclose(pt, tt, rtol=1e-2, atol=1e-5)
                        if not result:
                            print(
                                'diff max, min, mean: ',
                                (pt - tt).abs().max().item(),
                                (pt - tt).abs().min().item(),
                                (pt - tt).abs().mean().item(),
                            )

                            print(pt[(pt - tt).abs() > 1e-4])
                            print(tt[(pt - tt).abs() > 1e-4])
                            os.remove(out_path)
                            warnings.warn('The results don\'t match exactly')

                os.remove(out_path)

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
