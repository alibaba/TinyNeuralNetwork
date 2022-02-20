import re
import unittest

import torch
import torch.nn as nn

from tinynn.graph.tracer import fetch_modules, gen_module_constrctor_line


def collect_testcases():
    modules = fetch_modules()
    usages = []
    for mod in modules:
        name = mod.__name__
        if mod.__doc__ is not None:
            instances = re.findall(rf'nn\.{name}\(.*\)', mod.__doc__)
            usages.extend(list(set(instances)))
            if len(instances) == 0:
                print(f'{name} is skipped (no instances found)')
        else:
            print(f'{name} is skipped (doc missing)')

    results = []
    for usage in usages:
        try:
            m = eval(usage)
        except Exception:
            continue

        results.append((usage, m))
    return results


class TestModelMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_cases = collect_testcases()
        counter = dict()
        for usage, test_mod in test_cases:
            cls = type(test_mod)
            count = counter.get(cls, 0)
            count += 1
            test_name = f'test_{cls.__name__}_{count}'
            d[test_name] = mcls.build_model_test(usage, test_mod)
            counter[cls] = count
        return d

    @classmethod
    def build_model_test(cls, usage, test_mod):
        def f(self):
            line, _ = gen_module_constrctor_line(test_mod)
            try:
                eval(line)
            except Exception:
                self.fail(f'Cannot restore from {usage}, got {line}')

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
