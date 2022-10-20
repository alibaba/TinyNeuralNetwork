import torch
import torch.nn as nn
import torch.nn.quantized as nnq


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
        self.add1 = nnq.FloatFunctional()
        self.add2 = nnq.FloatFunctional()
        self.mul1 = nnq.FloatFunctional()
        self.mul2 = nnq.FloatFunctional()
        # self.div = torch.div()
        # self.sqrt = torch.sqrt()
        self.quant = torch.quantization.QuantStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # prelu(x) = relu(x) + weight * -relu(-x)
        # in which neg is implemented by multiplying an item with negative one
        # Also, we should use `add` and `mul` defined in `FloatFunctional`
        # What's more, both the weights and the negative one needs to go through
        # the QuantStub so as to make the whole computation graph quantized
        
        weight = self.weight.view(-1)
        bias = self.bias.view(-1)


        x1 = torch.var(input, dim=2, keepdim=True)
        x2 = torch.sqrt(x1)
        x3 = torch.mean(input)
        x4 = self.mul1.mul_scalar(x3, -1.0)
        x5 = self.add1.add(input, x4)
        x6 = torch.div(x5, x2)
        
        weight_q = self.quant(weight)
        bias_q = self.quant(bias)

        x7 = self.mul2.mul(weight_q, x6)
        x = self.add2.add_scalar(x7, bias_q)

        return x