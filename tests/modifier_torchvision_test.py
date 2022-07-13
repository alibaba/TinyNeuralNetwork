import time

import torch
import torchvision
import torchvision.models

import inspect
import os
import re
import gc
import unittest

from tinynn.graph.tracer import model_tracer

from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.util import import_from_path

from common_utils import collect_torchvision_models, prepare_inputs, IS_CI


def transform_output(output):
    new_output = []
    if type(output) in (list, tuple):
        for i in output:
            new_output.extend(transform_output(i))
    else:
        new_output.append(output.detach().numpy())
    return new_output


BLACKLIST = (
    'vit.*',
    'swin.*',
)

CI_BLACKLIST = (
    'convnext.*',
    'regnet_y_128gf',
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

            if IS_CI:
                for item in CI_BLACKLIST:
                    if re.match(item, model_name):
                        raise unittest.SkipTest('IN CI BLACKLIST')

            with model_tracer():
                m = model_class()
                m.eval()
                print(f'\n---------prune {model_name} ing!---------\n')

                inputs = prepare_inputs(m)
                outputs = m(*inputs)
                outputs = transform_output(outputs)

                # prune
                st = time.time()
                pruner = OneShotChannelPruner(m, inputs, {"sparsity": 0.5, "metrics": "l2_norm"})
                pruner.prune()

                pruner.graph.generate_code('out/new_model.py', 'out/new_model.pth', 'new_model')
                new_model = import_from_path('out.new_model', "out/new_model.py", "new_model")()
                new_model.load_state_dict(torch.load('out/new_model.pth'))

                print(f"[TEST] {model_name} cost {time.time() - st}")
                new_model(*inputs)

                # Remove the weights file to save space
                os.unlink('out/new_model.pth')

            if IS_CI:
                # Lower memory usage
                del inputs
                del outputs
                del pruner
                del new_model
                del m
                gc.collect()

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
