import torch
import torchvision

import os
import inspect

try:
    import models
except ImportError:
    models = None

IS_CI = os.getenv('CI', '') == 'true'


def collect_torchvision_models():
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
                elif p.name in ('pretrained', 'weights'):
                    has_pretrained = True
            if no_arg and has_pretrained:
                torchvision_model_classes.append(item)
    return torchvision_model_classes


def collect_custom_models():
    custom_model_classes = []
    if models is not None:
        for key in models.__dict__:
            item = getattr(models, key)

            if inspect.isclass(item) and issubclass(item, torch.nn.Module):
                if hasattr(item, '__module__'):
                    if item.__module__.startswith('torch.'):
                        continue

                if hasattr(item, '__init__') and hasattr(item, 'forward'):
                    constructor = getattr(item, '__init__')
                    no_arg = True
                    for p in inspect.signature(constructor).parameters.values():
                        if p.name not in ('self', 'kwargs') and p.default is p.empty:
                            no_arg = False
                            break
                    if no_arg:
                        custom_model_classes.append(item)
    return custom_model_classes


def prepare_inputs(model):
    if hasattr(model, 'custom_input_shape'):
        input_shape = model.custom_input_shape
    elif type(model).__name__ == 'inception_v3':
        input_shape = (1, 3, 299, 299)
    else:
        input_shape = (1, 3, 224, 224)

    if type(input_shape[0]) not in (tuple, list):
        input_shape = (input_shape,)

    inputs = []
    for shape in input_shape:
        t = torch.ones(shape)
        inputs.append(t)
    return inputs
