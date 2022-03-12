import gc
import inspect
import io
import logging
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models
from common_utils import (
    IS_CI,
    collect_custom_models,
    collect_torchvision_models,
    prepare_inputs,
)

from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer, trace


def show_source(model, title, reload_cache=True):
    source = inspect.getsource(type(model))
    lines = source.split('\n')
    start = lines[0].index('class')
    lines = [x[start:] for x in lines]
    source = '\n'.join(lines)

    print()
    print(title)
    print(source)


def check_quantize_rewrite(model, inputs, show_rewritten=True):
    with model_tracer():
        config = {'remove_weights_after_load': True}
        if sys.platform == 'win32':
            config.update({'backend': 'fbgemm', 'per_tensor': False})

        quantizer = QATQuantizer(model, inputs, work_dir='out', config=config)
        qat_model = quantizer.quantize()

        if show_rewritten:
            show_source(qat_model, 'Rewritten:')

        for _ in range(3):
            if isinstance(inputs, (list, tuple)):
                qat_model(*inputs)
            else:
                qat_model(inputs)

        with torch.no_grad():
            qat_model.eval()

            qat_model = torch.quantization.convert(qat_model)

            torch.jit.trace(qat_model, inputs)


class QuantizerTester(unittest.TestCase):
    def test_simple_float_model(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.reshape(3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_pow(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.pow(x, 2)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_sqrt(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.sqrt(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_sin(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.sin(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_cos(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.cos(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(not hasattr(F, 'hardsigmoid'), 'F.hardsigmoid not supported')
    def test_not_quantizable_hardsigmoid(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.hardsigmoid(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(not hasattr(nn, 'hardsigmoid'), 'nn.Hardsigmoid not supported')
    def test_not_quantizable_hardsigmoid_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activ = nn.Hardsigmoid()

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.activ(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(not hasattr(F, 'silu'), 'F.silu not supported')
    def test_not_quantizable_silu(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.silu(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(not hasattr(nn, 'SiLU'), 'nn.SiLU not supported')
    def test_not_quantizable_silu_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activ = nn.SiLU()

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.activ(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_reciprocal(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_exp(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.exp(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_softmax(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.softmax(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_softmax_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activ = nn.Softmax()

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.activ(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_log_softmax(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.log_softmax(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_log_softmax_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activ = nn.LogSoftmax()

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.activ(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_atan(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.atan(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_atan2(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return torch.atan2(x, y)

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.ones(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_truediv(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x / y

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.ones(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_layer_norm(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.randn(224, 224)
                self.b = torch.randn(224, 224)

            def forward(self, x):
                return F.layer_norm(x, (224, 224), self.w, self.b)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_layer_norm_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm((224, 224))

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.norm(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_instance_norm_module(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.instance_norm(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_instance_norm_1d_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.InstanceNorm1d(1)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.norm(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_instance_norm_2d_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.InstanceNorm2d(1)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.norm(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_rnn_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = torch.nn.RNN(224, 10, 1)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.rnn(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_lstm_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = torch.nn.LSTM(224, 10, 1)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.rnn(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_gru_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = torch.nn.GRU(224, 10, 1)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.rnn(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_avg_pool1d_with_one_kernel_size(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.avgpool = nn.AvgPool1d(1)

            def forward(self, x):
                return self.avgpool(x)

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_max_pool1d_with_one_kernel_size_and_stride(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = nn.MaxPool1d(1, 2)

            def forward(self, x):
                return self.maxpool(x)

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_avg_pool2d_with_one_kernel_size_with_stride(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 1, 2)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_max_pool2d_with_one_kernel_size(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 1)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_relu(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.relu(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_relu6(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.relu6(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_elu(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.elu(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_dropout(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.dropout(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_dropout_with_p(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.5)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_elu_with_alpha(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.elu(x, 1.5)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_neg(self):
        class Model(nn.Module):
            def forward(self, x):
                return -x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_div(self):
        class Model(nn.Module):
            def forward(self, x):
                return x / 2.0

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_sub(self):
        class Model(nn.Module):
            def forward(self, x):
                return x - 0.5

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_rsub(self):
        class Model(nn.Module):
            def forward(self, x):
                return 0.5 - x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_sub_tensors(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x - y

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_stack(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return torch.stack([x, y])

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_stack_with_dim(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return torch.stack([x, y], 1)

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_stack_with_dim_keyword(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return torch.stack([x, y], dim=1)

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_add(self):
        class Model(nn.Module):
            def forward(self, x):
                return x + 0.5

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_radd(self):
        class Model(nn.Module):
            def forward(self, x):
                return 0.5 + x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_add_tensors(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_mul(self):
        class Model(nn.Module):
            def forward(self, x):
                return x * 0.5

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_rmul(self):
        class Model(nn.Module):
            def forward(self, x):
                return 0.5 * x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_mul_tensors(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x * y

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_add_relu(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return (x + y).relu()

        model = Model()
        inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

        check_quantize_rewrite(model, inputs)

    def test_constant_pad_1d(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1))

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_constant_pad_2d(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1))

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_constant_pad_1d_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pad = nn.ConstantPad1d((1, 1), 0)

            def forward(self, x):
                return self.pad(x)

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_constant_pad_2d_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pad = nn.ConstantPad2d((1, 1), 0)

            def forward(self, x):
                return self.pad(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_zero_pad_2d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pad = nn.ZeroPad2d((1, 1))

            def forward(self, x):
                return self.pad(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_non_leaf_data(self):
        class Model(nn.Module):
            def forward(self, x):
                s = x.data
                s = s.shape
                return x.view(s)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)


if __name__ == '__main__':
    unittest.main()
