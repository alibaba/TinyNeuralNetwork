import math
from distutils.version import LooseVersion
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.intrinsic import ConvReLU1d
from torch.nn.modules.utils import _pair

from . import fused_modules as fm
from .utils import fuse_conv_bn_weights


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
        assert type(mod) is cls._FLOAT_MODULE, (
            'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if type(mod) is ConvReLU1d:
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
        assert type(mod) is cls._FLOAT_MODULE, (
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


class ConvTranspose2d(nn.ConvTranspose2d):
    r"""
    A ConvTranspose2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=convtranspose2d#torch.nn.ConvTranspose2d
    for documentation.

    Similar to `torch.nn.ConvTranspose2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.ConvTranspose2d

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
        super(ConvTranspose2d, self).__init__(
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

        return F.conv_transpose2d(
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
        assert type(mod) is cls._FLOAT_MODULE, (
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


_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

MOD = TypeVar('MOD', bound=nn.modules.conv._ConvTransposeNd)


class _ConvTransposeBnNd(nn.modules.conv._ConvTransposeNd, fm._FusedModule):
    _version = 2
    _FLOAT_MODULE = MOD

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        # BatchNormNd args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
        dim=2,
    ):
        nn.modules.conv._ConvTransposeNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            False,
            padding_mode,
        )
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self._enable_slow_path_for_better_numerical_stability = False

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        torch.nn.init.uniform_(self.bn.weight)
        torch.nn.init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ConvTransposeBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input):
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[1] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
        conv = self._conv_forward(input, scaled_weight, zero_bias)

        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvTransposeBnNd, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_ConvTransposeBnNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        assert type(mod) is cls._FLOAT_MODULE, (
            'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        )  # type: ignore[attr-defined]
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.output_padding,
            conv.groups,
            conv.bias is not None,
            conv.dilation,
            conv.padding_mode,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.bias is not None,
            self.dilation,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # fuse bn into conv
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
                True,
            )

        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv


class ConvTransposeBn2d(_ConvTransposeBnNd, nn.ConvTranspose2d):
    r"""
    A ConvTransposeBn2d module is a module fused from ConvTranspose2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.
    Similar to :class:`torch.nn.ConvTranspose2d`, with FakeQuantize modules initialized
    to default.
    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = fm.ConvTransposeBn2d
    _FLOAT_CONV_MODULE = nn.ConvTranspose2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = None

    def __init__(
        self,
        # ConvTransposeNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=None,
        dilation=1,
        padding_mode='zeros',
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvTransposeBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            output_padding,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=2,
        )

    def _conv_forward(self, input, weight, bias, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for _ConvTransposeNd')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )

    @classmethod
    def transform(cls, mod):
        conv = ConvTranspose2d(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            groups=mod.groups,
            bias=mod.bias is not None,
            dilation=mod.dilation,
            padding_mode=mod.padding_mode,
            qconfig=mod.qconfig,
        )

        conv.weight, conv.bias = fuse_conv_bn_weights(
            mod.weight,
            mod.bias,
            mod.bn.running_mean,
            mod.bn.running_var,
            mod.bn.eps,
            mod.bn.weight,
            mod.bn.bias,
            transpose=True,
        )

        conv.weight_fake_quant = mod.weight_fake_quant
        conv.activation_post_process = mod.activation_post_process

        return conv
