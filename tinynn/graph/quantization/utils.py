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


def fuse_bn_conv_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros(conv_w.shape[0], dtype=conv_w.dtype, device=conv_w.device)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    reduced_dims = [i for i in range(len(conv_w.shape)) if i > 1]

    fused_b = bn_b - bn_rm * bn_var_rsqrt * bn_w

    if conv_w.shape[1] == 1 and bn_rm.shape[0] > 1:
        offset_b = (conv_w.sum(dim=reduced_dims) * fused_b.reshape(-1, 1)).reshape(-1)
    else:
        offset_b = conv_w.sum(dim=reduced_dims).matmul(fused_b.reshape(-1, 1)).reshape(-1)

    fused_conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    fused_conv_b = conv_b + offset_b

    return torch.nn.Parameter(fused_conv_w, conv_w.requires_grad), torch.nn.Parameter(
        fused_conv_b, conv_b.requires_grad
    )


def get_parameter(mod: torch.nn.Module, param: str) -> torch.nn.Parameter:
    if param == 'weight':
        if isinstance(mod, torch.nn.Sequential):
            return getattr(mod[0], param)
        elif hasattr(mod, 'weight_fake_quant'):
            return mod.weight_fake_quant(getattr(mod, param))
        elif hasattr(mod, 'set_weight_bias'):
            return getattr(mod, param)()
        else:
            return getattr(mod, param)
    elif param == 'bias':
        if isinstance(mod, torch.nn.Sequential):
            return getattr(mod[0], param)
        elif hasattr(mod, 'weight_fake_quant'):
            return getattr(mod, param)
        elif hasattr(mod, 'set_weight_bias'):
            return getattr(mod, param)()
        else:
            return getattr(mod, param)
    else:
        return getattr(mod, param)


def clamp_with_fusion(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    if not x.is_quantized:
        return torch.clamp(x, min_val, max_val)
    return x


def clamp_with_fusion_(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    if not x.is_quantized:
        return torch.clamp_(x, min_val, max_val)
    return x


def fake_quantize(tensor, asym, eps, quant_max, quant_min):
    min_val, max_val = torch.aminmax(tensor)
    device = tensor.device
    zero_point = torch.zeros(min_val.size(), dtype=torch.int64, device=device)
    if not asym:
        max_val_pos = torch.max(-min_val, max_val)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = torch.max(scale, torch.tensor(eps))
    else:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = torch.max(scale, torch.tensor(eps))
        zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    # do fake quantize
    return torch.fake_quantize_per_tensor_affine(tensor, scale, zero_point, quant_min, quant_max)
