from distutils.version import LooseVersion

import numbers
from typing import Optional, Tuple, Union
import warnings

import torch
from torch import Tensor

from tinynn.util.train_util import get_logger

log = get_logger(__name__, 'WARNING')

if LooseVersion(torch.__version__) >= '1.13.0':

    @classmethod
    def from_float(cls, other, qconfig=None):
        assert isinstance(other, cls._FLOAT_MODULE)
        assert hasattr(other, 'qconfig') or qconfig
        observed = cls(
            other.input_size,
            other.hidden_size,
            other.num_layers,
            other.bias,
            other.batch_first,
            other.dropout,
            other.bidirectional,
        )
        observed.qconfig = getattr(other, 'qconfig', qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = _GRULayer.from_float(other, idx, qconfig, batch_first=False)
        observed.train()
        observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        return observed

    class GRUCell(torch.nn.Module):
        r"""A quantizable gated recurrent unit (GRU) cell.
        For the description and the argument types, please, refer to :class:`~torch.nn.GRUCell`
        """
        _FLOAT_MODULE = torch.nn.GRUCell

        def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.input_size = input_dim
            self.hidden_size = hidden_dim
            self.bias = bias

            self.igates = torch.nn.Linear(input_dim, 3 * hidden_dim, bias=bias, **factory_kwargs)
            self.hgates = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias, **factory_kwargs)

            self.add1 = torch.ao.nn.quantized.FloatFunctional()
            self.add2 = torch.ao.nn.quantized.FloatFunctional()
            self.add3 = torch.ao.nn.quantized.FloatFunctional()
            self.add4 = torch.ao.nn.quantized.FloatFunctional()
            self.sub1 = torch.ao.nn.quantized.FloatFunctional()

            self.mul1 = torch.ao.nn.quantized.FloatFunctional()
            self.mul2 = torch.ao.nn.quantized.FloatFunctional()
            self.mul3 = torch.ao.nn.quantized.FloatFunctional()
            self.mul4 = torch.ao.nn.quantized.FloatFunctional()

            self.act1 = torch.nn.Sigmoid()
            self.act2 = torch.nn.Tanh()
            self.hidden_state_dtype: torch.dtype = torch.quint8

        def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
            result = []
            if hidden is None or hidden[0] is None:
                hidden = self.initialize_hidden(x.shape[0], x.is_quantized)

            ri, zi, ni = self.igates(x).chunk(3, -1)

            if x.dim() > 2:
                for k in range(x.size(0)):
                    hx = hidden
                    rh, zh, nh = self.hgates(hx).chunk(3, -1)
                    rgate = self.act1(self.add1.add(ri[k, ...], rh))
                    zgate = self.act1(self.add2.add(zi[k, ...], zh))
                    ngate = self.act2(self.add3.add(ni[k, ...], self.mul1.mul(rgate, nh)))
                    hidden = self.add4.add(
                        self.mul2.mul(self.sub1.add_scalar(self.mul4.mul_scalar(zgate, -1), 1), ngate),
                        self.mul3.mul(zgate, hx),
                    )
                    result.append(hidden)
                result_tensor = torch.stack(result, 0)
                return result_tensor, hidden

            else:
                log.warning('Make sure you are not passing unbatched input to GRU, which may yield errors.')
                hx = hidden
                rh, zh, nh = self.hgates(hx).chunk(3, -1)
                rgate = self.act1(self.add1.add(ri, rh))
                zgate = self.act1(self.add2.add(zi, zh))
                ngate = self.act2(self.add3.add(ni, self.mul1.mul(rgate, nh)))
                hidden = self.add4.add(
                    self.mul2.mul(self.sub1.add_scalar(self.mul4.mul_scalar(zgate, -1), 1), ngate),
                    self.mul3.mul(zgate, hx),
                )
                result.append(hidden)
                result_tensor = torch.stack(result, 0)
                return result_tensor

        def initialize_hidden(self, batch_size: int, is_quantized: bool = False) -> Tensor:
            h = torch.zeros(batch_size, self.hidden_size)
            if is_quantized:
                (h_scale, h_zp) = self.initial_hidden_state_qparams
                h = torch.quantize_per_tensor(h, scale=h_scale, zero_point=h_zp, dtype=self.hidden_state_dtype)
            return h

        def _get_name(self):
            return 'QuantizableGRUCell'

        @classmethod
        def from_params(cls, wi, wh, bi=None, bh=None):
            """Uses the weights and biases to create a new GRU cell.
            Args:
                wi, wh: Weights for the input and hidden layers
                bi, bh: Biases for the input and hidden layers
            """
            assert (bi is None) == (bh is None)  # Either both None or both have values
            input_size = wi.shape[1]
            hidden_size = wh.shape[1]
            cell = cls(input_dim=input_size, hidden_dim=hidden_size, bias=(bi is not None))
            cell.igates.weight = torch.nn.Parameter(wi)
            if bi is not None:
                cell.igates.bias = torch.nn.Parameter(bi)
            cell.hgates.weight = torch.nn.Parameter(wh)
            if bh is not None:
                cell.hgates.bias = torch.nn.Parameter(bh)
            return cell

        @classmethod
        def from_float(cls, other):
            assert type(other) is cls._FLOAT_MODULE
            assert hasattr(other, 'qconfig'), "The float module must have 'qconfig'"
            observed = cls.from_params(other.weight_ih, other.weight_hh, other.bias_ih, other.bias_hh)
            observed.qconfig = other.qconfig
            observed.igates.qconfig = other.qconfig
            observed.hgates.qconfig = other.qconfig
            return observed

    class _GRUSingleLayer(torch.nn.Module):
        r"""A single one-directional GRU layer.
        The difference between a layer and a cell is that the layer can process a
        sequence, while the cell only expects an instantaneous value.
        """

        def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.cell = GRUCell(input_dim, hidden_dim, bias=bias, **factory_kwargs)

        def forward(self, x: Tensor, hidden: Optional[Tensor] = None):
            result_tensor, hidden = self.cell(x, hidden)
            return result_tensor, hidden

        @classmethod
        def from_params(cls, *args, **kwargs):
            cell = GRUCell.from_params(*args, **kwargs)
            layer = cls(cell.input_size, cell.hidden_size, cell.bias)
            layer.cell = cell
            return layer

    class _GRULayer(torch.nn.Module):
        r"""A single bi-directional GRU layer."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            bias: bool = True,
            batch_first: bool = False,
            bidirectional: bool = False,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.batch_first = batch_first
            self.bidirectional = bidirectional
            self.layer_fw = _GRUSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)
            if self.bidirectional:
                self.layer_bw = _GRUSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)

        def forward(self, x: Tensor, hidden: Optional[Tensor] = None):
            if self.batch_first:
                x = x.transpose(0, 1)

            hx_fw = hidden
            hidden_bw: Optional[Tensor] = None
            if self.bidirectional:
                if hx_fw is None:
                    hx_bw = None
                else:
                    hx_bw = hx_fw[1]
                    hx_fw = hx_fw[0]
                if hx_bw is not None:
                    hidden_bw = hx_bw

            if hx_fw is None:
                hidden_fw = None
            else:
                hidden_fw = torch.jit._unwrap_optional(hx_fw)
            result_fw, hidden_fw = self.layer_fw(x, hidden_fw)

            if hasattr(self, "layer_bw") and self.bidirectional:
                x_reversed = x.flip(0)
                result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
                result_bw = result_bw.flip(0)

                result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)

                if hidden_fw is None and hidden_bw is None:
                    h = None
                elif hidden_fw is None:
                    h = torch.jit._unwrap_optional(hidden_fw)
                elif hidden_bw is None:
                    h = torch.jit._unwrap_optional(hidden_bw)
                else:
                    h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)  # type: ignore[list-item]
            else:
                result = result_fw
                h = torch.jit._unwrap_optional(hidden_fw)  # type: ignore[assignment]

            if self.batch_first:
                result.transpose_(0, 1)

            return result, h

        @classmethod
        def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
            r"""
            There is no FP equivalent of this class. This function is here just to
            mimic the behavior of the `prepare` within the `torch.ao.quantization`
            flow.
            """
            assert hasattr(other, 'qconfig') or (qconfig is not None)

            input_size = kwargs.get('input_size', other.input_size)
            hidden_size = kwargs.get('hidden_size', other.hidden_size)
            bias = kwargs.get('bias', other.bias)
            batch_first = kwargs.get('batch_first', other.batch_first)
            bidirectional = kwargs.get('bidirectional', other.bidirectional)

            layer = cls(input_size, hidden_size, bias, batch_first, bidirectional)
            layer.qconfig = getattr(other, 'qconfig', qconfig)
            wi = getattr(other, f'weight_ih_l{layer_idx}')
            wh = getattr(other, f'weight_hh_l{layer_idx}')
            bi = getattr(other, f'bias_ih_l{layer_idx}', None)
            bh = getattr(other, f'bias_hh_l{layer_idx}', None)

            layer.layer_fw = _GRUSingleLayer.from_params(wi, wh, bi, bh)

            if other.bidirectional:
                wi = getattr(other, f'weight_ih_l{layer_idx}_reverse')
                wh = getattr(other, f'weight_hh_l{layer_idx}_reverse')
                bi = getattr(other, f'bias_ih_l{layer_idx}_reverse', None)
                bh = getattr(other, f'bias_hh_l{layer_idx}_reverse', None)
                layer.layer_bw = _GRUSingleLayer.from_params(wi, wh, bi, bh)
            return layer

    class GRU(torch.nn.Module):
        r"""A quantizable gated recurrent unit (GRU).
        For the description and the argument types, please, refer to :class:`~torch.nn.GRU`
        Attributes:
            layers : instances of the `_GRULayer`
        .. note::
            To access the weights and biases, you need to access them per layer.
            AssertionError: There is no reverse path in the non-bidirectional layer
        """
        _FLOAT_MODULE = torch.nn.GRU

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bias = bias
            self.batch_first = batch_first
            self.dropout = float(dropout)
            self.bidirectional = bidirectional
            self.training = False  # We don't want to train using this module

            if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
                raise ValueError(
                    "dropout should be a number in range [0, 1] representing the probability of an element being zeroed"
                )
            if dropout > 0:
                warnings.warn(
                    "dropout option for quantizable GRU is ignored. "
                    "If you are training, please, use nn.GRU version "
                    "followed by `prepare` step."
                )
                if num_layers == 1:
                    warnings.warn(
                        "dropout option adds dropout after all but last "
                        "recurrent layer, so non-zero dropout expects "
                        "num_layers greater than 1, but got dropout={} "
                        "and num_layers={}".format(dropout, num_layers)
                    )

            layers = [
                _GRULayer(
                    self.input_size,
                    self.hidden_size,
                    self.bias,
                    batch_first=False,
                    bidirectional=self.bidirectional,
                    **factory_kwargs,
                )
            ]
            for layer in range(1, num_layers):
                layers.append(
                    _GRULayer(
                        self.hidden_size,
                        self.hidden_size,
                        self.bias,
                        batch_first=False,
                        bidirectional=self.bidirectional,
                        **factory_kwargs,
                    )
                )
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, x: Tensor, hidden: Optional[Tensor] = None):
            if self.batch_first:
                x = x.transpose(0, 1)

            max_batch_size = x.size(1)
            num_directions = 2 if self.bidirectional else 1
            if hidden is None:
                zeros = torch.zeros(
                    num_directions, max_batch_size, self.hidden_size, dtype=torch.float, device=x.device
                )
                zeros.squeeze_(0)
                if x.is_quantized:
                    zeros = torch.quantize_per_tensor(zeros, scale=1.0, zero_point=0, dtype=x.dtype)
                hx = [zeros for _ in range(self.num_layers)]
            else:
                hidden_non_opt = torch.jit._unwrap_optional(hidden)
                if isinstance(hidden_non_opt, Tensor):
                    hx = hidden_non_opt.reshape(
                        self.num_layers, num_directions, max_batch_size, self.hidden_size
                    ).unbind(0)
                    hx = [(hx[idx].squeeze_(0)) for idx in range(self.num_layers)]
                elif isinstance(hidden_non_opt[0], Tensor):
                    hx = (
                        hidden_non_opt[0]
                        .reshape(self.num_layers, num_directions, max_batch_size, self.hidden_size)
                        .unbind(0)
                    )
                    hx = [(hx[idx].squeeze_(0)) for idx in range(self.num_layers)]
                else:
                    hx = hidden_non_opt

            hx_list = []
            for idx, layer in enumerate(self.layers):
                x, h = layer(x, hx[idx])
                hx_list.append(torch.jit._unwrap_optional(h))
            hx_tensor = torch.stack(hx_list)

            # We are creating another dimension for bidirectional case
            # need to collapse it
            hx_tensor = hx_tensor.reshape(-1, hx_tensor.shape[-2], hx_tensor.shape[-1])

            if self.batch_first:
                x = x.transpose(0, 1)

            return x, hx_tensor

        def _get_name(self):
            return 'QuantizableGRU'

        @classmethod
        def from_float(cls, other, qconfig=None):
            assert isinstance(other, cls._FLOAT_MODULE)
            assert hasattr(other, 'qconfig') or qconfig
            observed = cls(
                other.input_size,
                other.hidden_size,
                other.num_layers,
                other.bias,
                other.batch_first,
                other.dropout,
                other.bidirectional,
            )
            observed.qconfig = getattr(other, 'qconfig', qconfig)
            for idx in range(other.num_layers):
                observed.layers[idx] = _GRULayer.from_float(other, idx, qconfig, batch_first=False)
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
            return observed

        @classmethod
        def from_observed(cls, other):
            # The whole flow is float -> observed -> quantized
            # This class does float -> observed only
            raise NotImplementedError(
                "It looks like you are trying to convert a "
                "non-quantizable GRU module. Please, see "
                "the examples on quantizable GRUs."
            )
