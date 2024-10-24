import typing

import torch
import torch.nn as nn
import copy
from tinynn.util.util import get_logger

log = get_logger(__name__)

support_conv_cls = (torch.nn.Conv1d, torch.nn.Conv2d)
bn_restore_cls = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


class ConvBnForward(nn.Module):
    def __init__(self, conv):
        super().__init__()

        assert isinstance(conv, support_conv_cls), "not a supported conv type"
        self.conv = conv
        self.bn = bn_restore_cls[support_conv_cls.index(type(conv))](conv.out_channels)

    def forward(self, input_0):
        # Forward bn in train mode, but do not use its output so to get running_stat of conv at train_set.
        conv = self.conv(input_0)
        self.bn(conv)
        return conv


class ConvBnTrain(nn.Module):
    def __init__(self, convbn):
        super().__init__()
        assert isinstance(convbn, ConvBnForward), "not a ConvBnForward"
        self.conv = convbn.conv
        self.bn = convbn.bn
        # Use the forward running_stat to get weight and bias of restored bn, weight calculation ignores eps.
        weight = self.bn.running_var**0.5
        bias = self.bn.running_mean
        self.bn.weight.data.copy_(weight)
        self.bn.bias.data.copy_(bias)

    def forward(self, input_0):
        conv = self.conv(input_0)
        bn = self.bn(conv)
        return bn


def restore_bn(origin_model, layer_fused_bn):
    model = copy.deepcopy(origin_model)
    for name, mod in model.named_modules():
        if isinstance(mod, support_conv_cls) and name in layer_fused_bn:
            mod, mod_parent, name_part = get_submodule_with_parent_from_name(model, name)
            setattr(mod_parent, name_part, ConvBnForward(mod))
    return model


def restore_bn_set_param(origin_model):
    model = copy.deepcopy(origin_model)
    for name, mod in model.named_modules():
        if isinstance(mod, ConvBnForward):
            mod, mod_parent, name_part = get_submodule_with_parent_from_name(model, name)
            setattr(mod_parent, name_part, ConvBnTrain(mod))
    return model


def model_restore_bn(
    model: nn.Module, device, calibrate_func, *params, layers_fused_bn: typing.List[str] = None
) -> nn.Module:
    r"""High API to restore BN for a bn_fused model(e.g.MobileOne),
    the `calibrate_func` should be done in full train_set in train mode.

    Args:
        model (nn.Module): The model which need restore bn.
        device (torch.device): Specifies the device of the model.
        calibrate_func: The function used to do train_set calibrate in training mode.
        params: The params of calibrate function except model.
        layers_fused_bn: The name list of the conv which need to be bn_restored. Defaults to all conv of model.
    Return:
        The bn_restored model.

    """
    log.info("start to do BN Restore.")
    if layers_fused_bn is None:
        layers_fused_bn = [name for name, mod in model.named_modules() if isinstance(mod, support_conv_cls)]
    restore_bn_model = restore_bn(model, layers_fused_bn)
    restore_bn_model = restore_bn_model.train()
    restore_bn_model.to(device)
    calibrate_func(restore_bn_model, *params)
    bn_model = restore_bn_set_param(restore_bn_model)
    return bn_model


def get_submodule_with_parent_from_name(model, module_name):
    """Gets the submodule with its parent and sub_name using the name given"""
    module_name_parts = module_name.split('.')
    cur_obj = model
    last_obj = None

    for ns in module_name_parts:
        last_obj = cur_obj
        if type(cur_obj) is nn.ModuleList:
            cur_obj = cur_obj[int(ns)]
        elif type(cur_obj) is nn.ModuleDict:
            cur_obj = cur_obj[ns]
        else:
            cur_obj = getattr(cur_obj, ns)

    return cur_obj, last_obj, module_name_parts[-1]
