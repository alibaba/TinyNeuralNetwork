import torch
import torchvision
import torchvision.models

import inspect
import logging
import os
import re
import unittest

import numpy as np

from tinynn.graph.tracer import patch_helper, trace, tracer_context, model_tracer
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.util import import_from

from common_utils import collect_torchvision_models, prepare_inputs


def transform_output(output):
    new_output = []
    if type(output) in (list, tuple):
        for i in output:
            new_output.extend(transform_output(i))
    else:
        new_output.append(output.detach().numpy())
    return new_output


BLACKLIST = (
    'convnext.*',
    'vit.*',
    'swin.*'
)


class TestModelMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_classes = collect_torchvision_models()
        for test_class in test_classes:
            test_name = f'test_torchvision_model_{test_class.__name__}'
            d[test_name] = mcls.build_model_test(test_class)

        return d

    @classmethod
    def build_model_test(cls, model_class):
        def f(self):
            model_name = model_class.__name__

            for item in BLACKLIST:
                if re.match(item, model_name):
                    raise unittest.SkipTest('IN BLACKLIST')

            with model_tracer():
                m = model_class()
                m.eval()
                print(f'\n---------prune {model_name} ing!---------\n')

                inputs = prepare_inputs(m)
                outputs = m(*inputs)
                outputs = transform_output(outputs)

                # prune
                pruner = OneShotChannelPruner(m, inputs, {"sparsity": 0.75, "metrics": "l2_norm"})
                pruner.prune()
                outputs = m(*inputs)

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
