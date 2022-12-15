# 重参数类模型量化样例

## 背景

重参数模型（例如[RepVGG](https://arxiv.org/abs/2101.03697), [MobileOne](https://arxiv.org/abs/2206.04040)等）：这类模型在训练阶段构造一系列多分支结构，并在推理阶段将参数合并为单分支结构。训练阶段结构较大，可以得到更高的精度，而转换得到的推理模型结构较小且高精度可以保留。
例如在RepVGG中，训练阶段模型的一个block由3x3卷积、1x1卷积、Identity并联分支结构组成，在推理阶段则将这个block数学等价地变换为一个3x3的卷积。

由于重参数模型在部署推理模型时，需要将预训练模型中并联的多个卷积进行等价变换为一个卷积，其部署模型在量化时遇到了以下问题：
1. **权重离群点问题**：参数的等价变换导致不同的通道的卷积核的权重产生离群点，权重量化困难；这种现象在MobileOne的部署模型中尤为明显。
2. **BN缺失问题**：部分重参数类模型在部署时的等价变换过程中，卷积和BatchNorm(BN)融合为新的卷积，量化训练过程中没有BN层，训练不稳定，难以收敛。

## 解决方案

为了解决以上问题，我们给出了两个工具来近乎数学等价的修改预训练模型，以辅助量化。
1. [BatchNorm Restoration(BNR)](../../../tinynn/util/bn_restore.py)：BN恢复工具。由于重参数模型的推理模型中，BN已经被合并到卷积中，使用BNR工具恢复BN后可以有效助力量化。
2. [Cross_Layer_Equalization(CLE)](../../../tinynn/graph/quantization/cross_layer_equalization.py)：由[高通](https://arxiv.org/abs/1906.04721)提出。
   1. 使用CLE对预训练模型进行不同卷积权重之间的均匀化操作后，可以消除极大部分的权重离群点，极大降低了权重量化难度。CLE是数学等价的。
   2. 在进行CLE操作时，会将Conv-BN进行融合，此时模型的卷积带有bias。CLE会将部分卷积权重值较小的进行放大，同时导致bias放大，进而导致激活值放大，激活值量化困难；为了消除这种现象的影响，CLE完成后还会进行High Bias Absorb(HBA)操作将较大的bias吸收到后续的卷积中。

## 使用方法

1. 后量化：参考[`rep_ptq.py`](rep_ptq.py)。
2. 训练时量化：参考[`rep_qat.py`](rep_qat.py)。

## 实验结果(CIFAR10)：
| Model       | Top1 acc. (%) | Top1 acc. (%)<br/>(PTQ w/o CLE) | Top1 acc (%)<br/>(PTQ w/ CLE) |
|-------------|---------------|---------------------------------|-------------------------------|
| MobileOne   | 96.41         | 70.55(-25.86)                   | 95.44(-0.97)                  |
| RepVGG      | 94.46         | 46.35(-48.11)                   | 94.15(-0.31)                  |
| MobileNetV1 | 94.42         | 94.24(-0.18)                    | 94.30(-0.12)                  |

## 注意事项

1. BNR的接口为`tinynn.util.bn_restore.model_restore_bn`，给出部分参数说明和使用样例：
   1. `calibrate_func`：需要由用户给出，在训练集上以train模式进行完整的一个epoch推理。
   2. `layers_fused_bn`: 在进行BNR时，需要显式的给出需要进行BN恢复的卷积名称列表，默认为给定模型中的全部卷积算子。
   3. 使用样例：
```python
from tinynn.util.bn_restore import model_restore_bn

def calibare_func(model, context):
   model.train()
   for data in context.cali_loader:
      model(data)

device = troch.device('cpu') # 根据实际情况设置
context.cali_loader = train_dataloader # 设置对应的calibarte数据集
# 指定特定的卷积算子来进行BNR，否则默认会在所有卷积后都添加一个BN。
layers_fused_bn = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d) and 'reparam' in name]
model_restore_bn(model, device, calibare_func, context, layers_fused_bn=layers_fused_bn)
```
2. HBA操作需要预训练模型的卷积后有BN。由于重参数部署模型中BN已经被融合到卷积中，为了保证HBA的正常进行以提升量化精度，建议先进行BNR操作。
3. CLE的接口为`tinynn.graph.quantization.cross_layer_equalization.cross_layer_equalize`，给出部分参数说明：
   1. `threshold`：无效放大阻止阈值。CLE对weight进行均匀化调整时，某些卷积核的权重最大值过小（小于10e-23），均匀化时会将该通道的bias和权重放大10e10倍，导致激活值也放大10e10倍，HBA也无法消除这种放大。参数`threshold`可以有效避免这种无效放大，其默认值为1000，在CLE过程中，如果某个卷积放大幅度大于阈值时，将拒绝这次放大。若发现部分bias放大显著，则可以适当降低threshold。
   2. `cle_iters`: cle次数。单独进行一次CLE时，权重中可能还是存在部分离群点导致量化困难，可以进行多次CLE进一步提升量化精度。默认为2次。
4. 可选操作：在QAT之前，对重参数模型的部署模型进行CLE变换和BN重建之后，可以对浮点模型进行finetune训练，使模型量化训练更加稳定。
