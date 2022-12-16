import torch


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    fused_conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(fused_conv_w, conv_w.requires_grad), torch.nn.Parameter(
        fused_conv_b, conv_b.requires_grad
    )
