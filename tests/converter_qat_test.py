import torch
import torchvision
import torchvision.models

import io
import inspect
import logging
import os
import unittest

import numpy as np

from tinynn.converter import TFLiteConverter
from tinynn.graph.tracer import model_tracer, trace
from tinynn.graph.quantization.quantizer import QATQuantizer
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
        outputs.append(output_data)

    return outputs


# Unsupported models
# resnext: group convs
# yolov4: 5-d slices
BLACKLIST = ('resnext50_32x4d', 'resnext101_32x8d', 'Build_Model')


class TestModelMeta(type):

    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_classes = collect_torchvision_models()
        for test_class in test_classes:
            test_name = f'test_torchvision_model_{test_class.__name__}'
            simple_test_name = test_name + '_simple'
            d[simple_test_name] = mcls.build_model_test(test_class)
        # test_classes = collect_custom_models()
        # for test_class in test_classes:
        #     test_name = f'test_custom_model_{test_class.__name__}'
        #     simple_test_name = test_name + '_simple'
        #     d[simple_test_name] = mcls.build_model_test(test_class)
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

            args = ()
            kwargs = dict()
            if model_name in ('googlenet', 'inception_v3'):
                kwargs = {'aux_logits': False}

            with model_tracer():
                m = model_class(*args, **kwargs)
                m.cpu()
                m.eval()

                inputs = prepare_inputs(m)

                quantizer = QATQuantizer(m, inputs, work_dir='out', config={'remove_weights_after_load': True})
                qat_model = quantizer.quantize()

            with torch.no_grad():
                qat_model.eval()
                qat_model.cpu()

                qat_model = torch.quantization.convert(qat_model)

                out_path = f'out/{model_file}.tflite'
                converter = TFLiteConverter(qat_model, inputs, out_path)
                converter.convert()

                if HAS_TF:
                    outputs = converter.get_outputs()
                    input_transpose = converter.input_transpose
                    input_tf = data_to_tf(inputs, input_transpose)
                    tf_outputs = get_tflite_out(out_path, input_tf)
                    self.assertTrue(len(outputs) == len(tf_outputs))

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main(failfast=True)
