# 重参数类部署模型量化样例

## 背景

[重参数模型](https://arxiv.org/abs/2101.03697)（例如RepVgg, MobileOne等），首先在训练阶段构造一系列多分支结构，并在推理阶段将参数合并为单分支结构。训练阶段结构较大，可以得到更高的精度，而转换得到的推理模型结构较小且高精度可以保留。
例如在RepVgg中，训练阶段模型的一个block由3* 3卷积、1* 1卷积、identity并联分支结构组成，在推理阶段则将这个block数学等价地合并为一个3*3的卷积。

由于训练阶段的模型通过参数合并的方式得到了新的模型，部署模型在量化过程中存在以下问题：
1. 转换为部署模型的参数合并过程中，BatchNorm算子被合并到卷积中，量化训练时，精度损失严重，无法恢复浮点模型的精度。
2. 由于参数的合并导致，不同的通道的卷积核，尤其是对于DW卷积，其权重存在部分离群点（我们的实验中发现部分权重大于100），对权重进行per_tensor量化时,使用常用的MinMax量化时精度损失非常严重。

## 解决方案

为了解决以上两个问题，我们给出了两个工具来数学等价的修改预训练模型，以辅助量化。
1. BN_rebuild：由于重参数模型的推理模型中，BN已经被合并到卷积中，为了提升量化时训练的收敛性和稳定性（后量化不需要进行BN重建），我们实现了[BN恢复工具](../../../tinynn/util/bn_restore.py)。
2. Cross_Layer_Equalization（[CLE](https://arxiv.org/abs/1906.04721)）：由高通提出，用于对不同卷积之间的权重进行均衡化操作，可以很好的解决卷积中出现权重离群点的问题，我们实现了量化算法[Cross_Layer_Equalization](../../../tinynn/graph/quantization/cross_layer_equalization.py)，使用CLE后，可以极大的提升重参数类模型的量化精度。

## 使用方法

1. 后量化：参考`rep_ptq.py`。
2. 训练时量化：参考`rep_qat.py`。

## 注意事项

1. BN_rebuild的接口为`tinynn.util.bn_restore.model_restore_bn`，其需要提供一个在全训练集上以train mode进行校准操作的`calibrate_func`。此外，由参数`layers_fused_bn`指定需要进行bn重建的卷积算子的名称列表，默认为给定模型中的全部卷积算子。
2. Cross_Layer_Equalization为可选的操作，其接口为`tinynn.graph.quantization.cross_layer_equalization.cross_layer_equalize`。在我们的实验中，CLE可以很好的解决权重离群点的问题，有效提升后量化的精度以及保证训练时量化的初始精度。
3. CLE操作需要在BN_rebuild之前完成。
4. 在使用CLE的实际过程中，发现部分卷积的卷积核对应的特征输出会被放大10e8倍导致激活值量化失败，可以通过控制参数`threshold`来避免激活值量化失效。此外，单独进行一次CLE时，权重的均匀化程度有时还不能保证，可以尝试多次进行均匀化进一步提升量化精度。
5. 可选操作：在量化时训练准备阶段，对rep类模型进行CLE变换和BN重建之后，可以对浮点模型进行finetune训练，使模型量化训练更加稳定。
