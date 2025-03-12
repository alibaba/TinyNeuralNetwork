import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.quantization as torch_q


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


class QGLU(nn.Module):
    def __init__(self, glu: nn.GLU) -> None:
        super().__init__()

        self.dim = glu.dim
        self.f_mul = nnq.FloatFunctional()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(input, 2, self.dim)
        return self.f_mul.mul(slices[0], self.sigmoid(slices[1]))


class QHardsigmoid(nn.Module):
    def __init__(self, hardsigmoid: nn.Hardsigmoid) -> None:
        super().__init__()

        self.f_mul = nnq.FloatFunctional()
        self.f_add = nnq.FloatFunctional()
        self.q = torch_q.QuantStub()
        self.dq = torch_q.DeQuantStub()
        self.act_hs = nn.Hardsigmoid()
        self.act_r = nn.ReLU6()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.f_add.add_scalar(input, 3.0)
        x2 = self.act_r(x1)
        x3 = self.q(self.dq(x2))
        return self.f_mul.mul_scalar(x3, 1 / 6)


class QLayerNorm(nn.Module):
    def __init__(self, layernorm: nn.LayerNorm) -> None:
        super().__init__()
        self.mean_dims = tuple(range(-len(layernorm.normalized_shape), 0))
        self.weight = torch.nn.Parameter(layernorm.weight.data.detach().clone())
        self.bias = torch.nn.Parameter(layernorm.bias.data.detach().clone())
        self.eps = layernorm.eps

        self.q_rsqrt = torch_q.QuantStub()
        self.q_weight = torch_q.QuantStub()
        self.q_bias = torch_q.QuantStub()
        self.dq_rsqrt = torch_q.DeQuantStub()

        self.f_neg = nnq.FloatFunctional()
        self.f_add_0 = nnq.FloatFunctional()
        self.f_mul_0 = nnq.FloatFunctional()
        self.f_add_1 = nnq.FloatFunctional()
        self.f_mul_1 = nnq.FloatFunctional()
        self.f_mul_2 = nnq.FloatFunctional()
        self.f_add_2 = nnq.FloatFunctional()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # LayerNorm(input) = (input - mean(input)) * rsqrt(mean(( input - mean(input) )**20) + eps ) * alpha + beta
        # Currently, we completely split LayerNorm and independently count the quantization parameters
        # of the intermediate activation value, which may lead to a decrease in quantization accuracy.

        mean = input.mean(self.mean_dims, keepdim=True)
        diff = self.f_add_0.add(input, self.f_neg.mul_scalar(mean, -1.0).expand_as(input))
        squarer_difference = self.f_mul_0.mul(diff, diff)
        var = squarer_difference.mean(self.mean_dims, keepdim=True)
        var_eps = self.f_add_1.add_scalar(var, self.eps)

        fdq_var_eps = self.dq_rsqrt(var_eps)
        std_inverse = torch.rsqrt(fdq_var_eps)
        q_std_inverse = self.q_rsqrt(std_inverse)

        weight_fq = self.q_weight(self.weight)
        bias_fq = self.q_bias(self.bias)
        norm = self.f_mul_1.mul(diff, q_std_inverse)
        weight_fq_expand = weight_fq.expand_as(norm)
        norm_alpha = self.f_mul_2.mul(norm, weight_fq_expand)
        bias_fq_expand = bias_fq.expand_as(norm_alpha)
        return self.f_add_2.add(norm_alpha, bias_fq_expand)


class QRMSNorm(nn.Module):
    def __init__(self, rmsnorm: 'nn.RMSNorm') -> None:
        super().__init__()
        self.mean_dims = tuple(range(-len(rmsnorm.normalized_shape), 0))
        self.weight = torch.nn.Parameter(rmsnorm.weight.data.detach().clone())
        self.eps = rmsnorm.eps

        self.q_rsqrt = torch_q.QuantStub()
        self.q_weight = torch_q.QuantStub()
        self.dq_rsqrt = torch_q.DeQuantStub()

        self.f_add_0 = nnq.FloatFunctional()
        self.f_mul_0 = nnq.FloatFunctional()
        self.f_mul_1 = nnq.FloatFunctional()
        self.f_mul_2 = nnq.FloatFunctional()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # RMSNorm(input) = (input) * rsqrt(mean(input**2)) + eps ) * alpha + beta

        squard_input = self.f_mul_0.mul(input, input)
        if self.eps is None:
            rms_pre = squard_input.mean(self.mean_dims, keepdim=True)
        else:
            rms_pre = self.f_add_0.add_scalar(
                squard_input.mean(self.mean_dims, keepdim=True),
                self.eps,
            )

        fdq_rms_pre = self.dq_rsqrt(rms_pre)
        rms_inverse = torch.rsqrt(fdq_rms_pre)
        q_rms = self.q_rsqrt(rms_inverse)

        weight_fq = self.q_weight(self.weight)
        norm = self.f_mul_1.mul(input, q_rms)
        weight_fq_expand = weight_fq.expand_as(norm)
        return self.f_mul_2.mul(norm, weight_fq_expand)
