import torch.nn as nn
import torch.nn.intrinsic as nni

_FusedModule = getattr(nni, '_FusedModule', nn.Sequential)


class ConvTransposeBn2d(_FusedModule):
    def __init__(self, conv, bn):
        assert (
            type(conv) == nn.ConvTranspose2d and type(bn) == nn.BatchNorm2d
        ), 'Incorrect types for input modules{}{}'.format(type(conv), type(bn))
        super(ConvTransposeBn2d, self).__init__(conv, bn)


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

    if is_qat:
        return ConvTransposeBn2d(convt, bn)
    else:
        return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)
