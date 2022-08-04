import torch
from torch.nn import Module
from torch.quantization.observer import _with_args


class FakeQuantizeBFloat16(Module):
    """Simulate the quantize and dequantize operations in training time for bfloat16"""

    def __init__(self, **observer_kwargs):
        super(FakeQuantizeBFloat16, self).__init__()
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = 1 if enabled else 0

    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    def forward(self, X):
        if self.fake_quant_enabled[0] == 1:
            if isinstance(X, (tuple, list)):
                return [self.forward(x) for x in X]
            elif isinstance(X, torch.Tensor) and X.is_floating_point():
                dtype = X.dtype
                X = X.to(dtype=torch.bfloat16).to(dtype=dtype)
        return X

    with_args = classmethod(_with_args)


class FakeQuantizeTFLite(torch.quantization.FakeQuantize):
    def forward(self, X):
        observer_enabled = self.observer_enabled[0] == 1
        fake_quant_enabled = self.fake_quant_enabled[0] == 1

        if observer_enabled:
            if fake_quant_enabled:
                torch.quantization.disable_fake_quant(self)

            X = super().forward(X)

            if fake_quant_enabled:
                torch.quantization.enable_fake_quant(self)

        if fake_quant_enabled:
            if observer_enabled:
                torch.quantization.disable_observer(self)

            X = X + self.scale * 1e-6 * torch.sign(X.detach())
            X = super().forward(X)

            if observer_enabled:
                torch.quantization.enable_observer(self)

        return X


def disable_fake_quant(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::
      # model is any PyTorch model
      model.apply(tinynn.graph.quantization.disable_fake_quant)
    """
    if isinstance(mod, FakeQuantizeBFloat16):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    """
    Enable fake quantization for this module, if applicable. Example usage::
      # model is any PyTorch model
      model.apply(tinynn.graph.quantization.disable_fake_quant)
    """
    if isinstance(mod, FakeQuantizeBFloat16):
        mod.enable_fake_quant()
