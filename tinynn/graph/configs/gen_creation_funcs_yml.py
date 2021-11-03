import inspect
import torch
import yaml
import re
from torch.overrides import get_testing_overrides, get_overridable_functions

# TODO: Better detection
# The list is not complete, some functions are missing.

func_dict = get_overridable_functions()
final_dict = {'torch': []}

# Ignore the functions that we cannot translate (e.g. from_numpy)
block_list = ['from_numpy', 'frombuffer']

for k in torch.__dict__:
    if k in block_list:
        continue
    c = getattr(torch, k)
    if inspect.isclass(c) and k.endswith('Tensor') and c.__bases__[0] == object:
        print(k)
        final_dict['torch'].append(k)
    elif inspect.isbuiltin(c):
        if c not in func_dict[torch] and not k.startswith('_') and not k.endswith('_'):
            result_type = 'N/A'
            if c.__doc__:
                result_type = re.search(r'-> +(.*)', c.__doc__)
                if result_type:
                    result_type = result_type.group(1)
            else:
                if k.startswith('is_'):
                    result_type = 'bool [guess]'
                elif k.startswith('from_'):
                    result_type = 'Tensor'
                elif k.startswith('set_'):
                    result_type = 'None [guess]'

            if result_type and result_type.endswith('Tensor'):
                print(k, result_type)
                final_dict['torch'].append(k)

with open('torch_creation_funcs_override.yml', 'w') as f:
    yaml.dump(final_dict, f)
