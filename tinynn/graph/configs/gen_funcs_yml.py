import importlib
import inspect
import operator

import torch
import torchvision.ops
import yaml
from torch.overrides import get_ignored_functions, get_overridable_functions, get_testing_overrides

# TODO: Better detection

# Stage 1: Functions in get_overridable_functions()
func_dict = get_overridable_functions()
final_dict = {}

for k, v in func_dict.items():
    if not isinstance(k, type) and not inspect.ismodule(k):
        continue
    if type(v) is list:
        for vv in v:
            if hasattr(k, '__module__'):
                print(k.__module__, k.__name__, vv.__name__)
                final_dict.setdefault(f'{k.__module__}.{k.__name__}', [])
                if vv.__name__ not in final_dict[f'{k.__module__}.{k.__name__}']:
                    final_dict[f'{k.__module__}.{k.__name__}'].append(vv.__name__)
            elif hasattr(k, '__name__'):
                print(k.__name__, vv.__name__)
                final_dict.setdefault(k.__name__, [])
                if vv.__name__ not in final_dict[k.__name__]:
                    final_dict[k.__name__].append(vv.__name__)


# Stage 2: Functions in get_testing_overrides()
func_dict = get_testing_overrides()

for f, s in func_dict.items():
    qualname = None
    module = None
    name = None
    if hasattr(f, '__qualname__'):
        qualname = getattr(f, '__qualname__')
    if hasattr(f, '__module__'):
        module = getattr(f, '__module__')
    if hasattr(f, '__name__'):
        name = getattr(f, '__name__')
    if module and name:
        if module == 'torch._tensor':
            module = 'torch.Tensor'
            assert hasattr(torch.Tensor, name), f'{module}.{name}'
        elif module == 'torch._C._linalg':
            assert name.startswith('linalg_')
            module = 'torch.linalg'
            name = name.replace('linalg_', '')
            assert hasattr(torch.linalg, name), f'{module}.{name}'
        elif module == 'torch._C.nn':
            module = 'torch.nn'
            assert hasattr(torch.nn, name), f'{module}.{name}'
        elif module in ('torch._C.special', 'torch._C._special'):
            module = 'torch.special'
            name = name.replace('special_', '')
            assert hasattr(torch.special, name), f'{module}.{name}'
        elif module in ('torch._C.fft', 'torch._C._fft'):
            module = 'torch.fft'
            name = name.replace('fft_', '')
            assert hasattr(torch.fft, name), f'{module}.{name}'
        elif module == 'torch._C._nn':
            module = 'torch.nn.functional'
            if name == 'log_sigmoid':
                name = name.replace('_', '')
            assert hasattr(torch.nn.functional, name), f'{module}.{name}'
        elif module.startswith('torch._C.'):
            print(module, name, 'not recognized')
            assert False
        fullname = f'{module}.{name}'
    elif qualname:
        if qualname.startswith('_VariableFunctionsClass.'):
            fullname = qualname.replace('_VariableFunctionsClass.', 'torch.')
            assert fullname.count('.') == 1
            funcname = fullname.split('.')[1]
            assert hasattr(torch, funcname)
        elif qualname.startswith('torch._tensor.'):
            fullname = qualname.replace('torch._tensor.', 'torch.Tensor.')
            assert fullname.count('.') == 2
            funcname = fullname.split('.')[-1]
            assert hasattr(torch.Tensor, funcname)
        elif qualname.startswith('_TensorBase.'):
            fullname = qualname.replace('_TensorBase.', 'torch.Tensor.')
            assert fullname.count('.') == 2
            funcname = fullname.split('.')[-1]
            assert hasattr(torch.Tensor, funcname)
        else:
            pass

    print(fullname)
    rdot = fullname.rfind('.')

    ns = fullname[:rdot]
    func = fullname[rdot + 1 :]

    final_dict.setdefault(ns, [])
    final_dict[ns].append(func)

# Stage 3: Functions in get_ignored_functions() for the namespace (torch.nn.functional)
funcs = get_ignored_functions()
for f in funcs:
    qualname = None
    module = None
    name = None
    if hasattr(f, '__qualname__'):
        qualname = getattr(f, '__qualname__')
    if hasattr(f, '__module__'):
        module = getattr(f, '__module__')
    if hasattr(f, '__name__'):
        name = getattr(f, '__name__')

    if module == 'torch.nn.functional' and name is not None:
        if f.__doc__ is None:
            continue
        final_dict[module].append(name)

# Stage 4: torch.Tensor + operators
for k, v in operator.__dict__.items():
    if inspect.isbuiltin(v):
        if hasattr(torch.Tensor, k):
            final_dict['torch.Tensor'].append(k)

# Stage 5: torch.tensor -> torch.Tensor
if 'torch.tensor' in final_dict:
    v = final_dict.pop('torch.tensor')
    final_dict['torch.Tensor'].extend(v)

# Stage 6: torchvision ops
final_dict.setdefault('torchvision.ops', [])
for k, v in torchvision.ops.__dict__.items():
    if inspect.isroutine(v) and v.__doc__ is not None:
        final_dict['torchvision.ops'].append(k)


def get_scope(ns):
    spec = importlib.util.find_spec(ns)
    if spec is None:
        modules = ns.split('.')
        ns = '.'.join(modules[:-1])
        typename = modules[-1]
        spec = importlib.util.find_spec(ns)
        if spec is None:
            return None
        scope = importlib.import_module(ns)
        if hasattr(scope, typename):
            scope = getattr(scope, typename)
        else:
            return None
    else:
        scope = importlib.import_module(ns)

    return scope


ver = torch.__version__
ver = '_'.join(ver.split('.')[:2])

# Stage 7: Functions in new versions may exist in current version
latest = '1_11'
if ver != latest:
    with open(f'torch_func_override_{latest}.yml', 'r') as f:
        d = yaml.load(f, yaml.SafeLoader)
        for k, v in d.items():
            if k in final_dict:
                scope = get_scope(k)
                if scope is None:
                    continue

                for i in v:
                    if i not in final_dict[k]:
                        if hasattr(scope, i) and inspect.isroutine(getattr(scope, i)):
                            final_dict[k].append(i)
                            print(k, i)

# Stage 8: Functions may have different names (e.g. F.pad)
for k in final_dict:
    scope = get_scope(k)
    if scope is not None:
        for i in dir(scope):
            f = getattr(scope, i)
            if inspect.isroutine(f):
                if f.__name__ in final_dict[k] and i != f.__name__:
                    final_dict[k].append(i)
                    print(k, i)

# Stage 9: Make the functions unique and sorted
for k, v in final_dict.items():
    vv = list(set(v))
    vv.sort()
    v.clear()
    v.extend(vv)

# Stage 10: Update config
with open(f'torch_func_override_{ver}.yml', 'w') as f:
    yaml.dump(final_dict, f)
