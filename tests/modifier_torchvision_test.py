import torch
import torchvision
import torchvision.models

import inspect
import logging
import os
import unittest

import numpy as np

from tinynn.graph.tracer import patch_helper, trace, tracer_context, model_tracer
import models

from tinynn.prune.oneshot_pruner import OneShotChannelPruner


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def transform_output(output):
    new_output = []
    if type(output) in (list, tuple):
        for i in output:
            new_output.extend(transform_output(i))
    else:
        new_output.append(output.detach().numpy())
    return new_output


BLACKLIST = ()


class TestModelMeta(type):

    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_classes = mcls.collect_torchvision_models()
        for test_class in test_classes:
            test_name = f'test_torchvision_model_{test_class.__name__}'
            d[test_name] = mcls.build_model_test(test_class)

        return d

    @classmethod
    def build_model_test(cls, model_class):
        def f(self):
            model_name = model_class.__name__

            if model_name in BLACKLIST:
                raise unittest.SkipTest('IN BLACKLIST')

            with model_tracer():
                m = model_class()
                m.eval()
                print(f'\n---------prune {model_name} ing!---------\n')
                if hasattr(m, 'custom_input_shape'):
                    input_shape = m.custom_input_shape
                elif model_name == 'inception_v3':
                    input_shape = (1, 3, 299, 299)
                else:
                    input_shape = (1, 3, 224, 224)

                if type(input_shape[0]) not in (tuple, list):
                    input_shape = (input_shape, )

                inputs = []
                for shape in input_shape:
                    t = torch.ones(shape)
                    inputs.append(t)

                outputs = m(*inputs)
                outputs = transform_output(outputs)

                # prune
                pruner = OneShotChannelPruner(m, inputs, {"sparsity": 0.75, "metrics": "l2_norm"})
                pruner.prune()
                outputs = m(*inputs)

        return f

    @classmethod
    def collect_torchvision_models(cls):
        torchvision_model_classes = []
        for key in torchvision.models.__dict__:
            item = getattr(torchvision.models, key)
            if inspect.isfunction(item):
                no_arg = True
                has_pretrained = False
                for p in inspect.signature(item).parameters.values():
                    if p.name != 'kwargs' and p.default is p.empty:
                        no_arg = False
                        break
                    elif p.name == 'pretrained':
                        has_pretrained = True
                if no_arg and has_pretrained:
                    torchvision_model_classes.append(item)

        return torchvision_model_classes


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
