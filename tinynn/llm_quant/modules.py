import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.llama.modeling_llama import LlamaAttention
from tinynn.llm_quant.llama import LlamaAttentionFused
from tinynn.util.util import get_logger

from .util import _init_patch_easyquant, get_submodule_with_parent_from_name

log = get_logger(__name__, 'INFO')
SPEEDUP = True

try:
    if sys.platform == "win32":
        _init_patch_easyquant()

    from easyquant import (
        decompress_int4,
        decompress_int8,
        quantize_per_token,
        gemm,
        dequantize_bias_per_token,
        dequantize_per_token,
    )
except (ImportError, OSError):
    log.warning('easyquant is not installed, the inference performance may be degraded')
    SPEEDUP = False


def compress_int(data_tensor, bit_width, per_channel=True, per_token=False):
    # use [-127, 127] as 8-bit quant range
    q_max = 2 ** (bit_width - 1) - 1
    q_min = -q_max

    assert (per_channel and per_token) is False
    if per_channel:
        # for weight, use w_max/quant_max as scale, and convert weight to int8 to save memory.
        scale = 2 * (data_tensor.abs().max(dim=-1).values.float() / (2**bit_width - 1))
        quantized_tensor = torch.clamp(torch.round(data_tensor.float() / scale[:, None]), q_min, q_max).to(torch.int8)
    elif per_token:
        # per-token quantization
        scales = data_tensor.abs().max(dim=-1).values.float() / q_max
        if len(data_tensor.shape) == 3:
            scales = scales[:, :, None]
        elif len(data_tensor.shape) == 2:
            scales = scales[:, None]
        else:
            assert False
        quantized_tensor = torch.clamp(torch.round(data_tensor.float() / scales.float()), q_min, q_max).to(torch.int8)
        scale = scales
    else:
        # per_tensor quantization
        scale = data_tensor.abs().max().float() / q_max
        quantized_tensor = torch.clamp(torch.round(data_tensor.float() / scale.float()), q_min, q_max).to(torch.int8)

    return scale, quantized_tensor


class QLinear(nn.Module):
    def __init__(self, fc: nn.Linear, quant_mode: str):
        super().__init__()
        assert quant_mode in ("weight4", "weight8", "dynamic")
        if quant_mode == 'weight4':
            weight_bit_width = 4
        else:
            weight_bit_width = 8

        self.weight_bit_width = weight_bit_width
        self.quant_mod = quant_mode
        self.in_features = fc.in_features
        self.out_features = fc.out_features

        bias = None if fc.bias is None else fc.bias.data
        # compress weight by given bit, use per-channel and [-127,127]/[-7,7] to clamp
        scale, weight_q = compress_int(fc.weight.data, weight_bit_width)
        if self.in_features % 4 != 0 and quant_mode == 'dynamic':
            weight_q = F.pad(weight_q, (0, 4 - self.in_features % 4))

        if self.weight_bit_width == 4:
            weight_shape = weight_q.shape
            assert len(weight_shape) == 2
            assert weight_shape[1] % 2 == 0
            pre_packed = weight_q.view(weight_shape[0], weight_shape[1] // 2, 2)
            weight_q = ((pre_packed[..., 0] & 0b00001111) << 4) | (pre_packed[..., 1] & 0b00001111)

        self.weight = nn.Parameter(weight_q, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        fc.weight = None
        fc.bias = None

    def forward(self, input: Tensor) -> Tensor:
        input_device = input.device
        input_dtype = input.dtype
        input_shape = input.shape
        if self.quant_mod == 'static':
            assert False, f'{self.quant_mod} not supported'
        else:
            if self.quant_mod == 'weight4':
                if SPEEDUP:
                    weight_fp = torch.empty(
                        (self.out_features, self.in_features), dtype=torch.float16, device=input.device
                    )
                    decompress_int4(weight_fp, self.weight, self.weight_scale)
                else:
                    weight_fp = (
                        torch.stack((self.weight >> 4, self.weight << 4 >> 4), -1)
                        .view(self.out_features, self.in_features)
                        .to(dtype=torch.float32)
                        * self.weight_scale[:, None]
                    ).to(dtype=torch.half)
            elif self.quant_mod == 'weight8':
                if SPEEDUP:
                    weight_fp = torch.empty_like(self.weight.data, dtype=input_dtype, device=input_device)
                    decompress_int8(weight_fp, self.weight, self.weight_scale)
                else:
                    weight_fp = (self.weight.to(dtype=torch.float32) * self.weight_scale[:, None]).to(dtype=torch.half)

            if 'dynamic' in self.quant_mod:
                if SPEEDUP:
                    # the real dynamic quantization process, first quantize input to int8, then do int8Gemm calculation,
                    # and finally dequantize the output to float
                    input_viewed = input.view(-1, input_shape[-1])

                    # pad self.weight to 4x
                    padding_num = 4 - self.in_features % 4 if self.in_features % 4 != 0 else 0

                    # init easyquant kernels' output
                    input_q = torch.empty(
                        (input_viewed.shape[0], input_viewed.shape[1] + padding_num),
                        dtype=torch.int8,
                        device=input_device,
                    )
                    scale_shape = input_viewed.shape[0] if 'token' in self.quant_mod else 1
                    input_scale = torch.zeros(scale_shape, device=input_device)
                    out_q = torch.empty(
                        (int(input_viewed.shape[0]), self.out_features), dtype=torch.int32, device=input_device
                    )
                    output = torch.empty_like(out_q, dtype=torch.float16, device=input_device)

                    # use easyquant kernels to accelerate computation
                    quantize_per_token(input_q, input_viewed, input_scale)
                    gemm(out_q, input_q, self.weight)

                    if self.bias is not None:
                        dequantize_bias_per_token(output, out_q, input_scale, self.weight_scale, self.bias)
                    else:
                        dequantize_per_token(output, out_q, input_scale, self.weight_scale)

                    output = output.view(input_shape[:-1] + (output.shape[-1],))
                else:
                    # simulate quantization
                    input_scale, input_q = compress_int(input, 8, per_channel=False, per_token=True)
                    if self.in_features % 4 != 0:
                        output = F.linear(
                            input_q.float(), self.weight[:, : self.in_features % 4 - 4].float(), self.bias
                        )
                    else:
                        output = F.linear(input_q.float(), self.weight.float(), self.bias)
                    output = (output.float() * (self.weight_scale * input_scale.view(-1, 1))).half()
            else:
                input_fq = input
                output = F.linear(input_fq, weight_fp, self.bias)

        return output


class TDQLinear_noinit(QLinear):
    def forward(self, input: Tensor) -> Tensor:
        input_shape = input.shape
        bs, seq, _ = input_shape
        input_device = input.device
        input_viewed = input.view(-1, self.in_features)
        # pad self.weight to 4x
        padding_num = 4 - self.in_features % 4 if self.in_features % 4 != 0 else 0

        input_q = torch.empty(
            (input_viewed.shape[0], self.in_features + padding_num), dtype=torch.int8, device=input_device
        )
        input_scale = torch.empty(bs * seq, device=input_device)
        out_q = torch.empty((bs * seq, self.out_features), dtype=torch.int32, device=input_device)
        output = torch.empty_like(out_q, dtype=torch.float16, device=input_device)

        quantize_per_token(input_q, input_viewed, input_scale)
        gemm(out_q, input_q, self.weight)
        dequantize_per_token(output, out_q, input_scale, self.weight_scale)

        output = output.view(input_shape[:-1] + (output.shape[-1],))
        return output


@torch.no_grad()
def fuse_atten(model: nn.Module):
    """fuse qkv linear, fuse scaled_dot_product_attention if torch>=1.13"""
    for name, mod in model.named_modules():
        if isinstance(mod, LlamaAttention):
            _, parent_mod, last_name = get_submodule_with_parent_from_name(model, name)
            fused_attn = LlamaAttentionFused(mod)
            setattr(parent_mod, last_name, fused_attn)


@torch.no_grad()
def quant_fc(model: nn.Module, quant_mod='weight8', fuse_qkv=False):
    """convert all fcs of LLM model to quantized linear inplace.

    Args:
        model: the Given LLM model.
        quant_mod: the working quantization mode. Default to be 'weight8', Optional:['weight4', 'dynamic_token'].
            The 'dynamic_token' quantization use easyquant lib to do Int8Gemm accelerate.
        fuse_qkv: whether to fuse qkv linear of attention to speedup inference,
            the scaled-dot-product-attention will be fusedif the PyTorch version >= 1.13.
    """
    model.cpu()
    log.info(f'use quant mod {quant_mod} speedup={SPEEDUP}')
    if fuse_qkv:
        fuse_atten(model)
        log.info('qkv has been fused')

    for name, mod in model.named_modules():
        if 'lm_head' in name:
            continue
        if isinstance(mod, nn.Linear):
            _, parent_mod, last_name = get_submodule_with_parent_from_name(model, name)
            if quant_mod == 'dynamic' and SPEEDUP:
                quantized_fc_cls = TDQLinear_noinit
            else:
                quantized_fc_cls = QLinear
            quantized_fc = quantized_fc_cls(
                mod,
                quant_mod,
            )
            setattr(parent_mod, last_name, quantized_fc)
