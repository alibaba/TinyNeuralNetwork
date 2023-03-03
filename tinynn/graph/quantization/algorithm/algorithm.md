# Quantization Algorithm
[简体中文](algorithm_zh-CN.md)

1. [x] [Cross Layer Equalization](https://arxiv.org/abs/1906.04721)
2. [ ] [Bias Correction](https://arxiv.org/abs/1906.04721)
3. [ ] [Smooth Quant](https://arxiv.org/abs/2211.10438)

## Cross Layer Equalization, CLE

In some models, the weight distribution across channels varies greatly, and using a single quantization parameter to quantize all weights will result in significant quantization error.
When quantizing to 8 bits, for example, if one channel has weights in the range [128, 128] and another channel has weights in the range (0.5, 0.5), the weights in the latter channel will all be quantized to 0.

This type of weight has a "outlier phenomenon," which is more noticeable in models that heavily use DW convolution and reparameterization methods (such as MobileOne).

We implemented the cross-layer equalization (CLE) method in this repo by adopting [Qualcomm's] (https://arxiv.org/abs/1906.04721) cross-layer equalization algorithm.

```python
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize

model = cross_layer_equalize(model, dummy_input, device)
```

### Parameter
* model (nn.Module):

The origin model that requires CLE.
* dummy_input:

Dummy input for the creation of the TinyNN computation graph.
* device (torch.device):

Current device.
* threshold (float):

Weight range threshold for enabling CLE.

To calculate scales for a given channel $c_i$ and the weights of the neighboring 'Conv' layers $w_1$ and $w_2$, we use $scale=sqrt(\frac {\lvert max({w_1}_{c_i}) \rvert} {\lvert max({w_2}_{c_i}) \rvert})$.

When CLE is used, bias may occasionally be a concern.

For example, if the maximum value of ${w_1}_{c_i}$ is $2e-20$ and the maximum value of ${w_2}_{c_i}$ is $0.5$, the scale will be $1e-10$ and the bias of first conv will be scaled $1e10$ x.

To avoid this situation, a hyperparameter `threshold` is introduced, so that we disable CLE if $\lvert max({w_1}_{c_i}) \rvert + \lvert max({w_2}_{c_i}) \rvert < threshold$. Defaults to $0.5$.

* work_dir (str):

The working directory that stores the intermediate results of the CLE algorithm.
* cle_iters  (int):

The number of times CLE to be performed. The more times CLE is performed, the smoother the weight distribution.
However, according to the experimental results, a small number of CLE iterations is sufficient to achieve good quantization accuracy, whereas a large number of CLE iterations reduces accuracy.
The current default value is 2, and CLE should be repeated 1 to 5 times.
* hba_flag (bool):

Whether or not to perform the HBA operation. [Qualcomm](https://arxiv.org/abs/1906.04721) proposed HBA, which is used to eliminate the problem of bias amplification introduced by CLE at models with BatchNorm layers. Defaults to False.

### Application and Experiment Results
CLE can effectively address the issue of per-tensor quantization accuracy degradation caused by weight outliers.
Because of the reparameterization operations in specific model (e.g. RepVGG, MobileOne), the phenomenon of outliers is significant.
The following are the results of the experiments:

#### CIFAR10

| Model       | Top1 acc (%) | Top1 acc (%)<br/>(PTQ w/o CLE) | Top1 acc (%)<br/>(PTQ w/ CLE) |
|-------------|--------------|--------------------------------|-------------------------------|
| MobileOne   | 96.41        | 70.55(-25.86)                  | 95.44(-0.97)                  |
| RepVGG      | 94.46        | 46.35(-48.11)                  | 94.15(-0.31)                  |
| MobileNetV1 | 94.42        | 94.24(-0.18)                   | 94.30(-0.12)                  |
