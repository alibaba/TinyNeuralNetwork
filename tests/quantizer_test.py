import gc
import inspect
import io
import logging
import os
import sys
import unittest

from distutils.version import LooseVersion

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


def check_quantize_rewrite(model, inputs, show_rewritten=True, skip_train=False):
    with model_tracer():
        config = {'remove_weights_after_load': True}
        if sys.platform == 'win32':
            config.update({'backend': 'fbgemm', 'per_tensor': False})

        quantizer = QATQuantizer(model, inputs, work_dir='out', config=config)
        qat_model = quantizer.quantize()

        if show_rewritten:
            show_source(qat_model, 'Rewritten:')

        if not skip_train:
            for _ in range(3):
                if isinstance(inputs, (list, tuple)):
                    qat_model(*inputs)
                else:
                    qat_model(inputs)

        with torch.no_grad():
            qat_model.eval()

            qat_model = quantizer.convert(qat_model)

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

    @unittest.skipIf(not hasattr(nn, 'Hardsigmoid'), 'nn.Hardsigmoid not supported')
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

    def test_not_quantizable_abs(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.abs(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

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

        skip_train = False
        if LooseVersion(torch.__version__) >= LooseVersion('1.13.0'):
            skip_train = True

        check_quantize_rewrite(model, inputs, skip_train=skip_train)

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

    def test_tensor_div(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.div(2.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(
        LooseVersion(torch.__version__) < LooseVersion('1.7.0'), 'Integer division not supported in PyTorch 1.6'
    )
    def test_int_div(self):
        class Model(nn.Module):
            def forward(self, x):
                s = x.shape[-1] / 2
                x = x + s
                return x

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

    def test_add_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.c0 = torch.ones(1, 3, 224, 224)

            def forward(self, x):
                return x + self.c0

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_add_param_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter('c0', nn.Parameter(torch.ones(1, 3, 224, 224)))

            def forward(self, x):
                return x + self.c0

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

    def test_sum(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.sum(1)

        model = Model()
        inputs = torch.randn(1, 3, 224)

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

    def test_conv_relu6(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)
                self.activ = nn.ReLU6()

            def forward(self, x):
                s = self.conv(x)
                return self.activ(s)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_conv1d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return s

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(sys.platform == 'win32', "nnq.ConvTranspose1d is not available for FBGEMM")
    @unittest.skipIf(not hasattr(nn.quantized, 'ConvTranspose1d'), "nnq.ConvTranspose1d is not available")
    def test_conv_transpose1d(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.ConvTranspose1d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return s

        model = Model()
        inputs = torch.randn(1, 3, 224)

        check_quantize_rewrite(model, inputs)

    def test_fc_bn_rewrite(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(3072, 10)
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x):
                s_0 = x.view(x.shape[0], -1)
                s_1 = self.fc(s_0)
                return self.bn(s_1)

        model = Model()
        inputs = torch.randn(2, 3, 32, 32)

        check_quantize_rewrite(model, inputs)

    def test_fc_without_bias_bn_rewrite(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(3072, 10, bias=False)
                self.bn = nn.BatchNorm1d(10)

            def forward(self, x):
                s_0 = x.view(x.shape[0], -1)
                s_1 = self.fc(s_0)
                return self.bn(s_1)

        model = Model()
        inputs = torch.randn(2, 3, 32, 32)

        check_quantize_rewrite(model, inputs)

    @unittest.skipIf(LooseVersion(torch.__version__) < "1.10.0", "Quantization for F.embedding is not supported")
    def test_embedding(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.word_bank = torch.randn(10, 3)

            def forward(self, x):
                return F.embedding(x, self.word_bank)

        model = Model()
        inputs = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])

        check_quantize_rewrite(model, inputs)

    def test_embedding_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(10, 3)

            def forward(self, x):
                return self.embed(x)

        model = Model()
        inputs = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])

        check_quantize_rewrite(model, inputs)

    def test_float_cast(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = torch.arange(3)

            def forward(self, x):
                return x * self.c.reshape(1, -1, 1, 1).float()

        model = Model()
        inputs = torch.randn(1, 3, 1, 1)

        check_quantize_rewrite(model, inputs)

    def test_bmm_constant(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = torch.arange(3, dtype=torch.float32).view(1, 1, -1)

            def forward(self, x):
                return torch.bmm(x, self.c)

        model = Model()
        inputs = torch.randn(1, 3, 1)

        check_quantize_rewrite(model, inputs)

    def test_matmul_constant(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = torch.arange(3, dtype=torch.float32).view(1, 1, -1)

            def forward(self, x):
                return torch.bmm(x, self.c)

        model = Model()
        inputs = torch.randn(1, 3, 1)

        check_quantize_rewrite(model, inputs)

    def test_matmul_self(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.matmul(x, x)

        model = Model()
        inputs = torch.randn(1, 3, 3, 3)

        check_quantize_rewrite(model, inputs)

    def test_rsqrt(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.reciprocal(torch.sqrt(x))

        model = Model()
        inputs = torch.rand(1, 3, 3, 3) * 0.5 + 0.5

        check_quantize_rewrite(model, inputs)

    def test_rsqrt_1(self):
        class Model(nn.Module):
            def forward(self, x):
                return 1 / torch.sqrt(x)

        model = Model()
        inputs = torch.rand(1, 3, 3, 3) * 0.5 + 0.5

        check_quantize_rewrite(model, inputs)

    def test_rsqrt_1_with_prop(self):
        class Model(nn.Module):
            def forward(self, x):
                y = torch.sqrt(x)
                _ = y.dtype
                return 1 / y

        model = Model()
        inputs = torch.rand(1, 3, 3, 3) * 0.5 + 0.5

        check_quantize_rewrite(model, inputs)


if __name__ == '__main__':
    unittest.main()
