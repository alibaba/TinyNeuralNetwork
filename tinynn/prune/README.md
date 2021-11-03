# Model pruning
[简体中文](README_zh-CN.md)

## CIFAR10

#### Experiments on MobileNetV1-0.5 based on TinyNeuralNetwork

+ Sparsity: 0.5
+ Calculated FLOPs preserve rate: 0.25

+ Baseline accuracy: 94.2

| Plan                   | Acc.   |   ΔAcc.  |
| ---------------------- | ------ | -------- |
| OneShotPruner, random  | 94.04% | ↓ 0.16%  |
| OneShotPruner, l1_norm | 94.52% | ↑ 0.30%  |
| OneShotPruner, l2_norm | 94.59% | ↑ 0.37%  |
| OneShotPruner, fpgm    | 94.55% | ↑ 0.35%  |
| OneShotPruner, hrank   | 94.25% | ↑ 0.05%  |
| BNSlimingPruner        | 93.74% | ↓ 0.46%  |



#### Experiments on MobileNetV1-0.25 based on TinyNeuralNetwork

+ Sparsity: 0.25
+ Calculated FLOPs preserve rate: 0.06

+ Baseline accuracy: 94.2

| Plan                                       | Acc.   |   ΔAcc.  |
| ------------------------------------------ | ------ | -------- |
| OneShotPruner, random                      | 92.01% | ↓ 2.19%  |
| OneShotPruner, l1_norm                     | 93.36% | ↓ 0.84%  |
| OneShotPruner, l2_norm                     | 93.35% | ↓ 0.85%  |
| OneShotPruner, l2_norm, with distillation  | 93.48% | ↓ 0.72%  |
| OneShotPruner, hrank                       | 92.16% | ↓ 2.04%  |
| OneShotPruner, fpgm                        | 93.09% | ↓ 1.11%  |
| ADMMPruner, l2_norm                        | 93.56% | ↓ 0.64%  |
| GradualPruner, l2_norm                     | 93.78% | ↓ 0.42%  |
| RepPruner, l2_norm, with distillation      | 93.49% | ↓ 0.71%  |



## ImageNet

#### Comparsion of results on MobileNet-V1 between TinyNeuralNetwork and the original paper

Since the pre-trained model is not provided in the paper, we trained a model with similar accuracy and did pruning on top of that to minimize the error.
+ Accuracy in the paper: 70.5%
+ Accuracy of the baseline model used in TinyNeuralNetwork: 70.6%


| Model                                             | Acc.   |   ΔAcc.  |
| ------------------------------------------------- | ----- | -------- |
| MobileNetV1-1.0（Google）                          | 70.5% | —        |
| MobileNetV1-0.75（Google）                         | 68.4% | ↓ 2.1%   |
| MobileNetV1-0.75（TinyNeuralNetwork, OneShotPruner, l1_norm） | 70.1% | ↓ 0.4%   |
| MobileNetV1-0.5（Google）                          | 63.7% | ↓ 6.8%   |
| MobileNetV1-0.5（TinyNeuralNetwork, OneShotPruner, l1_norm）  | 64.7% | ↓ 5.7%   |
| MobileNetV1-0.5（TinyNeuralNetwork, GradualPruner, l2_norm）  | 65.6% | ↓ 4.9%   |

#### Comparsion of results on MobileNet-V1 between TinyNeuralNetwork and AMC

The [pretrained model](https://github.com/mit-han-lab/amc-models#download-the-pretrained-models) provided in the AMC repo is used as a baseline model to ensure fair comparison.
+ Baseline model accuracy: 71.4%
+ The models are trained 2 times using the AMC code and the TinyNeuralNetwork code respectively, and the highest accuracy is used for comparison

| Model                                             | Acc.   |   ΔAcc.  |
| ------------------------------------------------- | ------ | -------- |
| MobileNetV1-0.5（AMC）                            | 66.45% | ↓ 4.95%  |
| MobileNetV1-0.5（TinyNeuralNetwork, OneShotPruner, l1_norm） | 66.93% | ↓ 4.47%  |


