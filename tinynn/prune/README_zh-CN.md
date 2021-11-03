# 模型剪枝
[English](README.md)

## CIFAR10

#### TinyNeuralNetwork  MobileNetV1-0.5 部分方案对比

+ Channel 剩余 50%
+ Flops 剩余 25%

+ 剪枝前精度：94.2

| 方案                   | 精度   | 精度变化 |
| ---------------------- | ------ | -------- |
| OneShotPruner, random  | 94.04% | ↓ 0.16%  |
| OneShotPruner, l1_norm | 94.52% | ↑ 0.30%  |
| OneShotPruner, l2_norm | 94.59% | ↑ 0.37%  |
| OneShotPruner, fpgm    | 94.55% | ↑ 0.35%  |
| OneShotPruner, hrank   | 94.25% | ↑ 0.05%  |
| BNSlimingPruner        | 93.74% | ↓ 0.46%  |



#### TinyNeuralNetwork  MobileNetV1-0.25 部分方案对比

+ Channel 剩余 25%
+ Flops 剩余 6%

+ 剪枝前精度：94.2

| 方案                         | 精度   | 精度变化 |
| ---------------------------- | ------ | -------- |
| OneShotPruner, random        | 92.01% | ↓ 2.19%  |
| OneShotPruner, l1_norm       | 93.36% | ↓ 0.84%  |
| OneShotPruner, l2_norm       | 93.35% | ↓ 0.85%  |
| OneShotPruner, l2_norm, 蒸馏  | 93.48% | ↓ 0.72%  |
| OneShotPruner, hrank         | 92.16% | ↓ 2.04%  |
| OneShotPruner, fpgm          | 93.09% | ↓ 1.11%  |
| ADMMPruner, l2_norm          | 93.56% | ↓ 0.64%  |
| GradualPruner, l2_norm       | 93.78% | ↓ 0.42%  |
| RepPruner, l2_norm, 蒸馏      | 93.49% | ↓ 0.71%  |



## ImageNet

#### TinyNeuralNetwork 对比 MobileNetV1 论文数据

由于论文没有提供预训练模型，我们训练了一个精度接近的版本，并在此基础上做剪枝，尽量减少误差
+ 论文中精度：70.5%
+ 我们的精度：70.6%

| 模型                                               | 精度  | 精度损失 |
| -------------------------------------------------- | ----- | -------- |
| MobileNetV1-1.0（Google）                          | 70.5% | —        |
| MobileNetV1-0.75（Google）                         | 68.4% | ↓ 2.1%   |
| MobileNetV1-0.75（TinyNeuralNetwork, OneShotPruner, l1_norm） | 70.1% | ↓ 0.4%   |
| MobileNetV1-0.5（Google）                          | 63.7% | ↓ 6.8%   |
| MobileNetV1-0.5（TinyNeuralNetwork, OneShotPruner, l1_norm）  | 64.7% | ↓ 5.7%   |
| MobileNetV1-0.5（TinyNeuralNetwork, GradualPruner, l2_norm）  | 65.6% | ↓ 4.9%   |



#### TinyNeuralNetwork 对比 AMC 论文数据

确保效果对齐，采用AMC开源代码中提供的pretrain模型
+ 初始精度71.4%
+ 采用AMC代码与TinyNeuralNetwork代码分别训练2次，取精度最高值做对比

| 模型                                              | 精度   | 精度损失 |
| ------------------------------------------------- | ------ | -------- |
| MobileNetV1-0.5（AMC）                            | 66.45% | ↓ 4.95%  |
| MobileNetV1-0.5（TinyNeuralNetwork, OneShotPruner, l1_norm） | 66.93% | ↓ 4.47%  |


