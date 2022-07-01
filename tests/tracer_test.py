import torch
import torchvision
import torchvision.models

import inspect
import logging
import os
import re
import unittest

import numpy as np

from tinynn.graph.tracer import trace, model_tracer
from tinynn.util.util import import_from
from common_utils import collect_custom_models, collect_torchvision_models, prepare_inputs


BLACKLIST = (
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
            for eliminate_dead_graph in (False, True):
                simple_test_name = test_name + '_simple'
                if eliminate_dead_graph:
                    simple_test_name += '_edg'
                d[simple_test_name] = mcls.build_model_test(test_class, eliminate_dead_graph)

        test_classes = collect_custom_models()
        for test_class in test_classes:
            test_name = f'test_custom_model_{test_class.__name__}'
            for eliminate_dead_graph in (False, True):
                simple_test_name = test_name + '_simple'
                if eliminate_dead_graph:
                    simple_test_name += '_edg'
                d[simple_test_name] = mcls.build_model_test(test_class, eliminate_dead_graph)
        return d

    @classmethod
    def build_model_test(cls, model_class, eliminate_dead_graph):
        def f(self):
            model_name = model_class.__name__
            model_file = model_name
            model_file += '_simple'
            if eliminate_dead_graph:
                model_file += '_edg'

            for item in BLACKLIST:
                if re.match(item, model_name):
                    raise unittest.SkipTest('IN BLACKLIST')

            if os.path.exists(f'out/{model_file}.py'):
                raise unittest.SkipTest('TESTED')

            with model_tracer():
                m = model_class()
                m.eval()

                inputs = prepare_inputs(m)
                graph = trace(m, inputs, eliminate_dead_graph=eliminate_dead_graph)
                self.assertTrue(
                    graph.generate_code(f'out/{model_file}.py', f'out/{model_file}.pth', model_name, check=True)
                )

                # Remove the weights file to save space
                os.unlink(f'out/{model_file}.pth')

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
