from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.intrinsic import ConvReLU1d


class Conv1d(nn.Conv1d):
    r"""
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv1d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv1d#torch.nn.Conv1d
    for documentation.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        if LooseVersion(torch.__version__) >= LooseVersion('1.9.0'):
            factory_kwargs = {'device': device, 'dtype': dtype}
        else:
            factory_kwargs = {}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        if LooseVersion(torch.__version__) < LooseVersion('1.7.0'):
            self.activation_post_process = qconfig.activation()
        if LooseVersion(torch.__version__) >= LooseVersion('1.9.0'):
            self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)
        else:
            self.weight_fake_quant = qconfig.weight()

    def _conv_forward(self, input, weight, bias):
        if LooseVersion(torch.__version__) < '1.8.0':
            if self.padding_mode != 'zeros':
                return F.conv1d(
                    F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                    weight,
                    bias,
                    self.stride,
                    torch.nn.utils._single(0),
                    self.dilation,
                    self.groups,
                )
            return F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super()._conv_forward(input, weight, bias)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if type(mod) == ConvReLU1d:
            mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        conv = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        return conv


class ConvTranspose1d(nn.ConvTranspose1d):
    r"""
    A ConvTranspose1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose1d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=convtranspose1d#torch.nn.ConvTranspose1d
    for documentation.

    Similar to `torch.nn.ConvTranspose1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = 'zeros',
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        if LooseVersion(torch.__version__) >= LooseVersion('1.9.0'):
            factory_kwargs = {'device': device, 'dtype': dtype}
        else:
            factory_kwargs = {}
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        if LooseVersion(torch.__version__) < LooseVersion('1.7.0'):
            self.activation_post_process = qconfig.activation()
        if LooseVersion(torch.__version__) >= LooseVersion('1.9.0'):
            self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)
        else:
            self.weight_fake_quant = qconfig.weight()

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )

        return F.conv_transpose1d(
            input,
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            groups=mod.groups,
            bias=mod.bias is not None,
            dilation=mod.dilation,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv
