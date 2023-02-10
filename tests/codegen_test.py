import re
import unittest

import torch  # noqa: F401
import torch.nn as nn  # noqa: F401
import torch.nn.functional as F  # noqa: F401
import torchvision  # noqa: F401

from tinynn.graph.tracer import fetch_modules, gen_module_constrctor_line


def collect_testcases():
    modules = fetch_modules()
    usages = []
    for mod in modules:
        name = mod.__name__
        if mod.__doc__ is not None:
            full_instances = re.findall(
                rf'>>> .* = (nn.{name}\(.*\))\n *>>> input = (.*)\n *>>> output = .*?\((.*)\)', mod.__doc__
            )
            known_instances = set((full_instance[0] for full_instance in full_instances))
            instances = re.findall(rf'nn\.{name}\(.*\)', mod.__doc__)
            for instance in instances:
                if instance not in known_instances:
                    full_instances.append((instance, None, None))
            usages.extend(list(set(full_instances)))
            if len(full_instances) == 0:
                print(f'{name} is skipped (no instances found)')
        else:
            print(f'{name} is skipped (doc missing)')

    results = []
    for usage, dummy, args in usages:
        if dummy is not None and args is not None:
            content = f'm = {usage}\ninput = {dummy}\noutput = m({args})'
        else:
            content = f'm = {usage}'
        try:
            m = eval(usage)
            if dummy is not None and args is not None:
                input = eval(dummy)  # noqa: F841
                output = eval(f'm({args})')  # noqa: F841
        except Exception as e:
            print(f'{usage} is skipped: {content} errored with "{e}"')
            continue
        results.append((usage, dummy, args, m))
    return results


class TestModelMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        d = dict()
        test_cases = collect_testcases()
        counter = dict()
        for usage, dummy, args, test_mod in test_cases:
            cls = type(test_mod)
            count = counter.get(cls, 0)
            count += 1
            test_name = f'test_{cls.__name__}_{count}'
            d[test_name] = mcls.build_model_test(usage, dummy, args, test_mod)
            counter[cls] = count
        return d

    @classmethod
    def build_model_test(cls, usage, dummy, args, test_mod):
        def f(self):
            if dummy is not None and args is not None:
                content = f'm = {usage}\ninput = {dummy}\noutput = m({args})'
            else:
                content = f'm = {usage}'
            line, _ = gen_module_constrctor_line(test_mod)
            try:
                m = eval(line)  # noqa: F841
                if dummy is not None and args is not None:
                    input = eval(dummy)  # noqa: F841
                    output = eval(f'm({args})')  # noqa: F841
            except Exception:
                self.fail(f'Cannot restore from {content}, got {line}')

        return f


class TestModel(unittest.TestCase, metaclass=TestModelMeta):
    pass


if __name__ == '__main__':
    unittest.main()
