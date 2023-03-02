# 量化算法：
[English](algorithm.md)
1. [x] 跨层均衡算法([cross layer equalization](https://arxiv.org/abs/1906.04721))
2. [ ] 偏置修正算法([bias correction](https://arxiv.org/abs/1906.04721))
3. [ ] 激活值平滑算法([smooth quant](https://arxiv.org/abs/2211.10438))

## 跨层均衡算法（Corss Layer Equalization, CLE）

在某些模型中，其输出通道之间的权重分布存在极大差异，而对于per-tensor量化而言，使用单独一个量化参数来量化所有权重，会产生显著的量化误差。例如，如果一个通道的权重分布在[-128,128],而另一个通道分布在[-0.5,0.5]之间，那么在INT8量化时，后者将被全部量化为0
我们称之为这类权重存在"离群点现象"，尤其是在大量使用DW卷积和重参数方法的模型中(例如MobileOne)，离群点现象更加显著。

我们根据[Qualcomm](https://arxiv.org/abs/1906.04721)提出的跨层均衡算法，在TinyNeuralNetwork中集成了CLE算法：
```python
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize

model = cross_layer_equalize(model, dummy_input, device)
```

### 参数配置
* model (nn.Module):

需要进行CLE的原模型
* dummy_input:

模型示例输入，用于进行TinyNN的计算图生成
* device (torch.device):

当前使用的设备
* threshold (float):

不进行CLE的权重范围阈值，对于给定的通道，使用```scale=sqrt(abs_max(weight1_cn)/abs_max(weight2_cn))```计算缩放系数scale，
当weight1的最大值为2e-20，且weight2的最大值为0.5时，scale为1e-10，会将conv1对应的bias放大1e10倍，导致数值失效。
所以设置threshold，如果 ```abs_max(weight1_cn)+abs_max(weight2_cn) < threshold```，则将scale设置为1，不进行均衡操作。
默认设置为0.5。
* work_dir (str):

CLE算法中间结果的保存目录
* cle_iters  (int):

进行CLE的次数，CLE次数越多，权重分布将更加平滑，但是实验结果来看，cle_iters并非越大越好，当前默认值为2，推荐1~5次CLE即可。
* hba_flag (bool):

是否进行HBA操作，用于在带有BN的模型进行CLE之后，消除由CLE引入的bias放大的问题，默认为False关闭。

### 使用案例及实验结果
CLE可以很好地解决由于权重离群点产生的per-tensor量化精度下降问题。尤其是在重参数类模型中，由于其存在参数合并，导致离群点现象显著，以下为实验结果：
#### CIFAR10

| Model       | Top1 acc (%) | Top1 acc (%)<br/>(PTQ w/o CLE) | Top1 acc (%)<br/>(PTQ w/ CLE) |
|-------------|--------------|--------------------------------|-------------------------------|
| MobileOne   | 96.41        | 70.55(-25.86)                  | 95.44(-0.97)                  |
| RepVGG      | 94.46        | 46.35(-48.11)                  | 94.15(-0.31)                  |
| MobileNetV1 | 94.42        | 94.24(-0.18)                   | 94.30(-0.12)                  |
