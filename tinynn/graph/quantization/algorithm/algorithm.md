# Quantization Algorithm
[简体中文](algorithm_zh-CN.md)

1. [x] [cross layer equalization](https://arxiv.org/abs/1906.04721)
2. [ ] [bias correction](https://arxiv.org/abs/1906.04721)
3. [ ] [smooth quant](https://arxiv.org/abs/2211.10438)
## Corss Layer Equalization, CLE

In some models, the weight distribution across channels is very different, and for per-tensor quantization, using a single quantization parameter to quantize all weights will produce significant quantization error.
For example, in the case where one channel has weights in the range [−128, 128] and another channel has weights in the range (−0.5, 0.5),
the weights in the latter channel will all be quantized to 0 when quantizing to 8-bits.

We call this type of weights have "outlier phenomenon", especially in models that heavily use DW convolution and heavy parameter methods (such as MobileOne), the outlier phenomenon is more significant.

We followed the cross-layer-equalization algorithm proposed by [Qualcomm](https://arxiv.org/abs/1906.04721), and integrated the CLE algorithm in TinyNeuralNetwork:
```python
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize

model = cross_layer_equalize(model, dummy_input, device)
```

### Parameter
* model (nn.Module):

The origin model which need to do CLE.
* dummy_input:

Dummy input for TinyNN calculation graph generation.
* device (torch.device):

Current device.
* threshold (float):

Weight range threshold without CLE. For given channel, we use ```scale=sqrt(abs_max(weight1_cn)/abs_max(weight2_cn))``` to calculate scales,
in some case ,the maximum value of weight1 maybe very small like 2e-20 while the maximum value of weight2 can be 0.5, the scale will be 1e-10, and the bias of first conv will be scaled 1e10x,
so we set threshold, if ```abs_max(weight1_cn)+abs_max(weight2_cn) < threshold```, we set this channel's scale to 1 to prevent the scale factor becoming too large. Default to be 0.5.

* work_dir (str):

The work directory of the intermediate results of the CLE algorithm.
* cle_iters  (int):

The number of times to perform CLE, the more times of CLE, the smoother the weight distribution will be.
However, according to the experimental results, small cle_iters is enough to get good quantization acc while too many cle iter will decrease acc.
The current default value is 2, and 1 to 5 times of CLE is recommended.
* hba_flag (bool):

Whether to perform HBA operation, which is used to eliminate the problem of bias amplification introduced by CLE at model with BatchNorm. Default to be False.

### Usage and Experimental Results
CLE can well solve the problem of per-tensor quantization accuracy degradation due to weight outliers.
Especially in the Reparameter-style model(e.g. RepVgg, MobileOne), due to the reparameterization of parameters, the phenomenon of outliers is significant.
The following are the experimental results:
#### CIFAR10

| Model       | Top1 acc (%) | Top1 acc (%)<br/>(PTQ w/o CLE) | Top1 acc (%)<br/>(PTQ w/ CLE) |
|-------------|--------------|--------------------------------|-------------------------------|
| MobileOne   | 96.41        | 70.55(-25.86)                  | 95.44(-0.97)                  |
| RepVGG      | 94.46        | 46.35(-48.11)                  | 94.15(-0.31)                  |
| MobileNetV1 | 94.42        | 94.24(-0.18)                   | 94.30(-0.12)                  |
