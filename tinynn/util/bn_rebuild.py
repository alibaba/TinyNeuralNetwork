import typing

import torch
import torch.nn as nn
import copy


class ConvBnForward(nn.Module):
    def __init__(self, conv):
        super().__init__()

        assert isinstance(conv, nn.Conv2d), "not a conv2d"
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels)

    def forward(self, input_0):
        # We forward bn in train mode, but do not use its output to get running stat in train_set.
        conv = self.conv(input_0)
        self.bn(conv)
        return conv


class ConvBnTrain(nn.Module):
    def __init__(self, convbn):
        super().__init__()
        assert isinstance(convbn, ConvBnForward), "not a ConvBnForward"
        self.conv = convbn.conv
        self.bn = convbn.bn
        eps = self.bn.eps
        # Use the forward running_stat to get weight and bias of rebuilt bn.
        weight = (self.bn.running_var + eps) ** 0.5
        bias = self.bn.running_mean
        self.bn.weight.data.copy_(weight)
        self.bn.bias.data.copy_(bias)

    def forward(self, input_0):
        conv = self.conv(input_0)
        bn = self.bn(conv)
        return bn


def add_bn(origin_model, layer_fused_bn):
    model = copy.deepcopy(origin_model)
    for name, mod in model.named_children():
        if isinstance(mod, nn.Conv2d) and name in layer_fused_bn:
            setattr(model, name, ConvBnForward(mod))
    return model


def add_bn_set_param(origin_model):
    model = copy.deepcopy(origin_model)
    for name, mod in model.named_children():
        if isinstance(mod, ConvBnForward):
            setattr(model, name, ConvBnTrain(mod))
    return model


def model_add_bn(model: nn.Module, layers_fused_bn: typing.List[str], device, calibrate, *params):
    """high API to rebuild BN for a bn_fused model(e.g.mobileone)

    Args:
        model (nn.Module): The model which need rebuild bn.
        layers_fused_bn (typing.List[str]): The list of layers which need rebuild bn.
        device (torch.device): Specifies the device of the model.
        calibrate: The function used to do train_set calibrate in training mode.
        params: The params of calibrate function except model.

    Return:
        The bn_rebuilt model.
    """
    add_bn_model = add_bn(model, layers_fused_bn)
    add_bn_model = add_bn_model.train()
    add_bn_model.to(device)
    calibrate(add_bn_model, *params)
    bn_model = add_bn_set_param(add_bn_model)
    return bn_model
