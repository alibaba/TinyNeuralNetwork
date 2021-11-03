import torch
import inspect
import yaml

final_dict = {}

# Stage 1: Quantization stubs
for k in torch.quantization.stubs.__dir__():
    c = getattr(torch.quantization.stubs, k)
    if not isinstance(c, type) and not inspect.ismodule(c):
        continue
    if isinstance(c, type):
        print(k, c, issubclass(c, torch.nn.Module))
        print(c.__module__)
        final_dict.setdefault('torch.quantization', [])
        final_dict['torch.quantization'].append(k)

# Stage 2: torch.nn Modules
for k in torch.nn.__dict__:
    c = getattr(torch.nn, k)
    if not isinstance(c, type) and not inspect.ismodule(c):
        continue
    if isinstance(c, type):
        print(k, c, issubclass(c, torch.nn.Module))
        print(c.__module__)
        # Skip container modules
        if '.container' in c.__module__:
            continue
        if c.__name__ in ('Parameter', 'Module'):
            continue
        final_dict.setdefault('torch.nn', [])
        final_dict['torch.nn'].append(k)

# Stage 3: Update config
with open('torch_module_override.yml', 'w') as f:
    yaml.dump(final_dict, f)
