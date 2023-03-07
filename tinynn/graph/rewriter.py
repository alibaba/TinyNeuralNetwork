import torch
import torch.nn as nn


def gen_layernorm(mod):
    class CustomLayerNorm(torch.autograd.Function):
        @staticmethod
        def symbolic(g, input):
            return g.op(
                "trt::LayerNorm",
                input,
                g.op("Constant", value_t=mod.weight.data),
                g.op("Constant", value_t=mod.bias.data),
                epsilon_f=mod.eps,
                axis_i=-len(mod.normalized_shape),
            )

        @staticmethod
        def forward(ctx, x):
            return torch.nn.functional.layer_norm(x, mod.normalized_shape, mod.weight.data, mod.bias.data, mod.eps)

    return CustomLayerNorm


MOD_DICT = {nn.LayerNorm: gen_layernorm}


def gen_rewrite_hook(gen):
    def rewrite_hook(mod, inp, outp):
        return gen(mod).apply(inp[0])

    return rewrite_hook


def rewrite_for_tensorrt_export(model):
    for m in model.modules():
        gen = MOD_DICT.get(type(m), None)
        if gen is not None:
            m.register_forward_hook(gen_rewrite_hook(gen))
