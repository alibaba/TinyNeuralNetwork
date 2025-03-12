import inspect
import sys
import unittest

from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinynn.graph.quantization.quantizer import DeQuantizer, QATQuantizer, PostQuantizer
from tinynn.graph.tracer import model_tracer


def show_source(model, title, reload_cache=True):
    source = inspect.getsource(type(model))
    lines = source.split('\n')
    start = lines[0].index('class')
    lines = [x[start:] for x in lines]
    source = '\n'.join(lines)

    print()
    print(title)
    print(source)


def check_quantize_rewrite(model, inputs, show_rewritten=True, skip_train=False, skip_trace=False, is_qat=True):
    with model_tracer():
        config = {'remove_weights_after_load': True, 'ignore_layerwise_config': True}
        if sys.platform == 'win32':
            config.update({'backend': 'fbgemm', 'per_tensor': False})

        if is_qat:
            quantizer_cls = QATQuantizer
        else:
            quantizer_cls = PostQuantizer

        quantizer = quantizer_cls(model, inputs, work_dir='out', config=config)
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

            # Evaluation on converted model doesn't fail
            if isinstance(inputs, (list, tuple)):
                qat_model(*inputs)
            else:
                qat_model(inputs)

            if not skip_trace:
                torch.jit.trace(qat_model, inputs)


def check_dequantize_rewrite(model, inputs, show_rewritten=True, skip_train=False):
    with model_tracer():
        config = {'remove_weights_after_load': True}

        dequantizer = DeQuantizer(model, inputs, work_dir='out', config=config)
        float_model = dequantizer.dequantize()

        if show_rewritten:
            show_source(float_model, 'Rewritten:')

        if not skip_train:
            for _ in range(1):
                if isinstance(inputs, (list, tuple)):
                    float_model(*inputs)
                else:
                    float_model(inputs)


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

    def test_not_quantizable_prelu(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.zeros(3).fill_(0.25)

            def forward(self, x):
                return F.prelu(x, self.weight)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_prelu_weight(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(3).fill_(0.25))

            def forward(self, x):
                return F.prelu(x, self.weight)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_prelu_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activ = nn.PReLU()

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
        inputs = torch.randn(1, 3, 224, 224) * 0.5 + 0.5

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

    def test_not_quantizable_group_norm(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.randn(4)
                self.b = torch.randn(4)

            def forward(self, x):
                return F.group_norm(x, 2, self.w, self.b)

        model = Model()
        inputs = torch.randn(1, 4, 224, 224)

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
        torch.nn.init.uniform_(model.norm.bias, -0.1, 0.1)
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

    def test_not_quantizable_group_norm_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.GroupNorm(2, 4)

            def forward(self, x):
                return self.norm(x)

        model = Model()
        inputs = torch.randn(1, 4, 224, 224)

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
                self.rnn = torch.nn.GRU(224, 10, 2, bidirectional=True)

            def forward(self, x):
                y = torch.split(x, 1, 1)
                return self.rnn(y[0])

        model = Model()
        inputs = torch.randn(1, 3, 224)

        skip_train = False
        if LooseVersion(torch.__version__) >= LooseVersion('1.13.0'):
            skip_train = True

        check_quantize_rewrite(model, inputs, skip_train=skip_train)

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

    def test_functional_to_module_sigmoid(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.sigmoid(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_tanh(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.tanh(x)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_functional_to_module_hardswish(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.hardswish(x)

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

    def test_stack_with_shared_tensors(self):
        class Model(nn.Module):
            def forward(self, x):
                tensors = x.split(1, 1)
                return torch.stack(tensors)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

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

    def test_stack_with_axis_keyword(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return torch.stack([x, y], axis=1)

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

    def test_conv_clamp(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return torch.clamp(s, 0.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_conv_clamp_1(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return torch.clamp(s, 0.0, 6.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_conv_clamp_2(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return torch.clamp(s, -6.0, 6.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224) * 10

        check_quantize_rewrite(model, inputs)

    def test_conv_clamp_min(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x):
                s = self.conv(x)
                return torch.clamp_min(s, 0.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_clamp_max(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.clamp_max(x, 1.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_single_clamp(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.clamp(x, -1.0, 1.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_single_clamp_min_only(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.clamp(x, -1.0)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_single_clamp_max_only(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.clamp(x, None, -1.0)

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

    def test_not_quantizable_glu(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.glu(x)

        model = Model()
        inputs = torch.rand(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_not_quantizable_glu_module(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.act = nn.GLU()

            def forward(self, x):
                return self.act(x)

        model = Model()
        inputs = torch.rand(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_multiple_inputs_from_same_op(self):
        class Model(nn.Module):
            def forward(self, x):
                x = x.relu().unbind(1)
                x = (x[0] / x[1]).relu()
                return x

        model = Model()
        inputs = torch.rand(1, 3, 224, 224).abs() * 0.5 + 0.5

        check_quantize_rewrite(model, inputs)

    def test_quantized_add_different_shape(self):
        class Model(nn.Module):
            def forward(self, x):
                return x + x[:, 0:1]

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_quantized_add_different_shape_complex(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.squeeze(0) + x[:, 0:1]

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_quantized_mul_different_shape(self):
        class Model(nn.Module):
            def forward(self, x):
                return x * x[:, 0:1]

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_quantized_mul_different_shape_complex(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1) * x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_quantized_add_relu_different_shape(self):
        class Model(nn.Module):
            def forward(self, x):
                return (x + x[:, 0:1]).relu()

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_known_param_from_module_weight(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return x + torch.squeeze(self.conv.weight, 1).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_known_param_from_module_bias(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return x + self.conv.bias.view(-1, 1, 1).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_known_param_from_module_weight_qmod(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return self.conv(x).expand(1, 3, 224, 224) + torch.squeeze(self.conv.weight, 1).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, skip_trace=True)

    def test_known_param_from_module_weight_qmod_reverse(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return torch.squeeze(self.conv.weight, 1).expand(1, 3, 224, 224) + self.conv(x).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, skip_trace=True)

    def test_known_param_from_module_bias_qmod(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return self.conv(x).expand(1, 3, 224, 224) + self.conv.bias.view(-1, 1, 1).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, skip_trace=True)

    def test_known_param_from_module_bias_qmod_reverse(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1, 1)

            def forward(self, x):
                return self.conv.bias.view(-1, 1, 1).expand(1, 3, 224, 224) + self.conv(x).expand(1, 3, 224, 224)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, skip_trace=True)

    def test_known_param_simple(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = nn.Parameter(torch.randn(1, 3, 224, 224))

            def forward(self, x):
                return x * self.param

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_known_param_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = nn.Parameter(torch.randn(1, 3, 224, 224))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub_module = SubModule()

            def forward(self, x):
                return x * self.sub_module.param

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs)

    def test_type_conversion_to_int_tensor(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = torch.zeros((1, 256), dtype=torch.int64)

            def forward(self, x):
                return x.to(self.data)

        model = Model()
        inputs = torch.randn(1, 3, 256, 256)

        check_quantize_rewrite(model, inputs)

    def test_type_conversion_to_int_dtype(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.data = torch.zeros((1, 256), dtype=torch.int64)

            def forward(self, x):
                return x.to(self.data.dtype)

        model = Model()
        inputs = torch.randn(1, 3, 256, 256)

        check_quantize_rewrite(model, inputs)

    def test_bn_conv_fusion(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn(x)
                x = self.conv(x)
                return x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, is_qat=False)

    def test_bn_conv_relu_fusion(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)
                self.bn = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.bn(x)
                x = self.conv(x)
                x = self.relu(x)
                return x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_quantize_rewrite(model, inputs, is_qat=False)


class DeQuantizerTester(unittest.TestCase):
    def test_simple_q_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()

            def forward(self, x):
                return self.dq(self.q(x).reshape(3, 224, 224))

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_add_relu(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                x_r = x.reshape(3, 224, 224)
                return self.f.add_relu(x, x_r)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_add(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                x_r = x.reshape(3, 224, 224)
                return self.f.add(x, x_r)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_mul(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                x_r = x.reshape(3, 224, 224)
                return self.f.mul(x, x_r)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_cat(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                return self.f.cat([x, x], -1)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_add_scalar(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                return self.f.add_scalar(x, 0.5)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_simple_q_mul_scalar(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.q = torch.quantization.QuantStub()
                self.dq = torch.quantization.DeQuantStub()
                self.f = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.q(x)
                return self.f.mul_scalar(x, 0.5)

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)

    def test_conv_transpose_bn_fusion(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose2d(3, 5, 2, 2, 1)
                self.bn = nn.BatchNorm2d(5)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = Model()
        inputs = torch.randn(1, 3, 224, 224)

        check_dequantize_rewrite(model, inputs)


if __name__ == '__main__':
    unittest.main()
