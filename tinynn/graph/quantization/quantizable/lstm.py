from distutils.version import LooseVersion

import torch

if LooseVersion(torch.__version__) >= '1.13.0':

    from torch.ao.nn.quantizable.modules.rnn import _LSTMLayer

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
            observed.layers[idx] = _LSTMLayer.from_float(other, idx, qconfig, batch_first=False)
        observed.train()
        observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        return observed
