import copy
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni

from .utils import fuse_conv_bn_weights, fuse_bn_conv_weights

_FusedModule = getattr(nni, '_FusedModule', nn.Sequential)


class ConvTransposeBn2d(_FusedModule):
    def __init__(self, conv, bn):
        assert (
            type(conv) is nn.ConvTranspose2d and type(bn) is nn.BatchNorm2d
        ), 'Incorrect types for input modules{}{}'.format(type(conv), type(bn))
        super(ConvTransposeBn2d, self).__init__(conv, bn)


HAS_QAT_IN_FUNC = LooseVersion(torch.__version__) >= '1.11.0'


def fuse_convtranspose_bn(is_qat, convt, bn):
    r"""Given ConvTranspose and bn modules, fuses them and returns the fused module

    Args:
        convt: Module instance of type ConvTransposeNd
        bn: BatchNormNd instance that needs to be fused with the linear layer.
            batch norm N should match the ConvTranspose N

    Examples::

        >>> m1 = nn.ConvTranspose2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_convtranspose_bn(m1, b1)
    """
    assert convt.training == bn.training, "ConvTranspose and BN both must be in the same mode (train or eval)."

    if is_qat is None:
        is_qat = convt.training

    if is_qat:
        return ConvTransposeBn2d(convt, bn)
    else:
        if HAS_QAT_IN_FUNC:
            return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)
        else:
            fused_conv = copy.deepcopy(convt)

            fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, True
            )

            return fused_conv


def fuse_bn_conv(is_qat, bn, conv):
    assert conv.training == bn.training, "Conv and BN both must be in the same mode (train or eval)."

    if is_qat is None:
        is_qat = conv.training

    assert not is_qat, "BN Conv fusion for QAT is not yet supported"

    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_bn_conv_weights(
        fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )

    return fused_conv


def fuse_bn_conv_relu(is_qat, bn, conv, relu):
    assert conv.training == bn.training == relu.training, "Conv, BN and ReLU must be in the same mode (train or eval)."

    if is_qat is None:
        is_qat = conv.training

    assert not is_qat, "BN Conv fusion for QAT is not yet supported"

    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_bn_conv_weights(
        fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )

    return nni.ConvReLU2d(fused_conv, relu)


PATCH_MOD_MAPPING = {
    (
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
    ): fuse_convtranspose_bn,
    (
        nn.BatchNorm2d,
        nn.Conv2d,
    ): fuse_bn_conv,
    (
        nn.BatchNorm2d,
        nn.Conv2d,
        nn.ReLU,
    ): fuse_bn_conv_relu,
}


def gen_fuse_known_modules_wrapper(orig_fuse_known_modules):
    def fuse_known_modules(mod_list, *args, **kwargs):
        types = tuple(type_before_parametrizations(m) for m in mod_list)
        if HAS_QAT_IN_FUNC:
            is_qat = kwargs.get('is_qat', args[0])
        else:
            is_qat = None
        if types in PATCH_MOD_MAPPING:
            new_mod = [None] * len(mod_list)
            fuse_func = PATCH_MOD_MAPPING[types]
            fused = fuse_func(is_qat, *mod_list)
            # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
            # Move pre forward hooks of the base module to resulting fused module
            for handle_id, pre_hook_fn in mod_list[0]._forward_pre_hooks.items():
                fused.register_forward_pre_hook(pre_hook_fn)
                del mod_list[0]._forward_pre_hooks[handle_id]
            # Move post forward hooks of the last module to resulting fused module
            for handle_id, hook_fn in mod_list[-1]._forward_hooks.items():
                fused.register_forward_hook(hook_fn)
                del mod_list[-1]._forward_hooks[handle_id]
            new_mod[0] = fused

            for i in range(1, len(mod_list)):
                identity = nn.Identity()
                identity.training = mod_list[0].training
                new_mod[i] = identity

            return new_mod
        else:
            return orig_fuse_known_modules(mod_list, *args, **kwargs)

    return fuse_known_modules


def type_before_parametrizations(module):
    has_parametrize = hasattr(torch.nn.utils, 'parametrize')
    has_is_parametrized = has_parametrize and hasattr(torch.nn.utils.parametrize, 'is_parametrized')
    has_type_before_parametrizations = has_parametrize and hasattr(
        torch.nn.utils.parametrize, 'type_before_parametrizations'
    )

    if has_type_before_parametrizations:
        return torch.nn.utils.parametrize.type_before_parametrizations(module)
    elif has_is_parametrized and torch.nn.utils.parametrize.is_parametrized(module):
        return module.__class__.__bases__[0]
    else:
        return type(module)
