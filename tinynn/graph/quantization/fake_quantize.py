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


class PTQFakeQuantize(torch.quantization.FakeQuantize):
    """Using fake-quantize to do PTQ, speed up quantization error analyze.
    When doing calibrate, please enable observer and disable fake-quant,
    when validating model with quantization error, please enable fake-quant and disable observer.
    """

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())

        if self.fake_quant_enabled[0] == 1:
            if (self.scale == 1 and self.zero_point == 0) or (
                float(self.scale * (self.quant_max - self.quant_min + 1)) == 256
            ):
                _scale, _zero_point = self.calculate_qparams()
                _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max
                )
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max
                )
        return X


def set_ptq_fake_quantize(name, module):
    weight_fq = PTQFakeQuantize.with_args(
        observer=torch.quantization.MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    asym_fq = PTQFakeQuantize.with_args(
        observer=torch.quantization.HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        reduce_range=False,
    )
    qconfig_new = torch.quantization.QConfig(asym_fq, weight_fq)
    return qconfig_new
