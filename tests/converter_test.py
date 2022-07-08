import torch
import torchvision
import torchvision.models

import gc
import io
import inspect
import logging
import os
import re
import unittest
import warnings

import numpy as np

from tinynn.converter import TFLiteConverter
from common_utils import IS_CI, collect_custom_models, collect_torchvision_models, prepare_inputs


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


def get_tflite_out(model_path, inputs, output_transpose):
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
        if output_transpose[i]:
            output_data = np.transpose(output_data, [0, 3, 1, 2])
        outputs.append(output_data)

    return outputs


# Unsupported models
# resnext: group convs
# regnet: group convs
# yolov4: 5-d slices
# swin: roll
# vit: native_multi_head_attention
BLACKLIST = (
    'resnext.*',
    'Build_Model',
    'regnet.*',
    'swin.*',
    'vit.*',
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

            for item in BLACKLIST:
                if re.match(item, model_name):
                    raise unittest.SkipTest('IN BLACKLIST')

            if os.path.exists(f'out/{model_file}.tflite'):
                raise unittest.SkipTest('TESTED')

            m = model_class()
            m.eval()

            inputs = prepare_inputs(m)

            with torch.no_grad():
                out_path = f'out/{model_file}.tflite'

                extra_kwargs = {}
                if IS_CI:
                    out_pt = f'out/{model_file}.pt'
                    extra_kwargs.update({'dump_jit_model_path': out_pt, 'gc_when_reload': True})

                converter = TFLiteConverter(m, inputs, out_path, **extra_kwargs)

                if IS_CI:
                    # Remove original model to lower memory usage
                    del m

                converter.convert()

                if IS_CI:
                    os.remove(out_pt)

                if HAS_TF:
                    outputs = converter.get_outputs()
                    input_transpose = converter.input_transpose
                    output_transpose = converter.output_transpose
                    input_tf = data_to_tf(inputs, input_transpose)
                    tf_outputs = get_tflite_out(out_path, input_tf, output_transpose)
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

                if IS_CI and os.path.exists(out_path):
                    os.remove(out_path)

            if IS_CI:
                # Lower memory usage
                del converter
                gc.collect()

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
