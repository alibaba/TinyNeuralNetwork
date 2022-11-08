from cv2 import meanShift
from pyparsing import quotedString
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import numpy as np
import math

class QPReLU(nn.Module):
    def __init__(self, prelu: nn.PReLU) -> None:
        super().__init__()

        # Copies weight from existing PReLU object
        self.weight = torch.nn.Parameter(prelu.weight.data.detach().clone())

        # Other necessary modules for QAT preparation
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.f_mul_neg_one1 = nnq.FloatFunctional()
        self.f_mul_neg_one2 = nnq.FloatFunctional()
        self.f_mul_alpha = nnq.FloatFunctional()
        self.f_add = nnq.FloatFunctional()
        self.quant = torch.quantization.QuantStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # prelu(x) = relu(x) + weight * -relu(-x)
        # in which neg is implemented by multiplying an item with negative one
        # Also, we should use `add` and `mul` defined in `FloatFunctional`
        # What's more, both the weights and the negative one needs to go through
        # the QuantStub so as to make the whole computation graph quantized
        x1 = self.relu1(input)

        if self.weight.numel() == 1:
            weight = self.weight.view(())
        else:
            weight_shape = self.weight.shape + (1,) * (input.dim() - 2)
            weight = self.weight.view(*weight_shape)

        weight_q = self.quant(weight)
        x2 = self.f_mul_alpha.mul(
            weight_q,
            self.f_mul_neg_one2.mul_scalar(
                self.relu2(
                    self.f_mul_neg_one1.mul_scalar(input, -1.0),
                ),
                -1.0,
            ),
        )

        x = self.f_add.add(x1, x2)
        return x


class QSiLU(nn.Module):
    def __init__(self, _: 'nn.SiLU') -> None:
        super().__init__()

        self.act = nn.Sigmoid()
        self.f_mul = nnq.FloatFunctional()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.f_mul.mul(input, self.act(input))


class QLayerNorm(nn.Module):
    def __init__(self, layernorm: nn.LayerNorm) -> None:
        super().__init__()

        # Copies weight and bias from existing LayerNorm object
        self.weight = torch.nn.Parameter(layernorm.weight.data.detach().clone())
        self.bias = torch.nn.Parameter(layernorm.bias.data.detach().clone())

        # Other necessary modules for QAT preparation
        # self.mean = torch.mean()
        # self.var = torch.var()
        self.add = nnq.FloatFunctional()
        self.mul = nnq.FloatFunctional()
        self.div = nnq.FloatFunctional()
        # self.div = torch.div()
        # self.sqrt = torch.sqrt()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def qsrt(self, a):
        # x = torch.pow(2, torch.ceil(torch.tensor(int.bit_length(a)/2)))
        # x = a/2
        # print(x)
        
        a1 = self.dequant(a)
        x = a1
        b = torch.div(a1, x, rounding_mode='trunc')
        x2 = torch.div(b, 2, rounding_mode='trunc')

        b = torch.div(a1, x2, rounding_mode='trunc')
        x3 = torch.div(b, 2, rounding_mode='trunc')

        b = torch.div(a1, x3, rounding_mode='trunc')
        x4 = torch.div(b, 2, rounding_mode='trunc')

        b = torch.div(a1, x4, rounding_mode='trunc')
        x5 = torch.div(b, 2, rounding_mode='trunc')
        x = self.quant(x5)
        # print(x)
        # x = torch.floor((x+torch.floor(a/x))/2)
        return x

    def loop(self, input, x):
        dq_input = self.dequant(input)
        x1 = self.dequant(x)
        first = torch.div(dq_input, x1)
        q_first = self.quant(first)
        second = self.add.add(q_first, x)
        dq_second = self.dequant(second)
        third = torch.div(dq_second, 2)
        q_third = self.quant(third)  
        return q_third


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # eps = 1e-5
        # mean = x.mean(dim=-1, keepdim=True)
        # # print(mean)
        # var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # # print(var)
        # std = (var + eps).sqrt()
        # y = (x - mean) / std
        # print(y-output_layernorm)
        weight = self.weight.view(-1)
        bias = self.bias.view(-1)
        weight_q = self.quant(weight)
        bias_q = self.quant(bias)

        mean = torch.mean(input, dim=-1, keepdim=True)
        mean_opp = self.mul.mul_scalar(mean, -1.0)
        mean_sub = self.add.add(mean, mean_opp)
        mean_mul = self.mul.mul(mean_sub, mean_sub)
        var = torch.mean(mean_mul, dim=-1, keepdim=True)
        var_add = self.add.add_scalar(var, 1)
        # x1 = torch.div(input, var_add)
        # loop1 = self.loop(var_add, var_add)
        # loop2 = self.loop(loop1, var_add)
        # loop3 = self.loop(loop2, var_add)
        # std = self.loop(loop3, var_add)
        std = self.qsrt(var_add)
        std1 = self.dequant(std)
        mean_sub1 = self.dequant(mean_sub)
        div1 = mean_sub1/std1
        div = self.quant(div1)
        x1 = self.mul.mul(div, weight_q)
        x = self.add.add(x1, bias_q)

        return x