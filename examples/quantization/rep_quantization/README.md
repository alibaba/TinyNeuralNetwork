# Rep_deploy model Quantization
[简体中文](README_zh-CN.md)

## Background

RepVGG-like model(e.g. [RepVGG](https://arxiv.org/abs/2101.03697), [MobileOne](https://arxiv.org/abs/2206.04040)): Such models construct a series of multi-branch structures during training and merge the parameters into a single-branch structure during the inference.
For example, a block of the RepVGG in the training phase is a parallel branch structure consisting of a 3x3 convolution, a 1x1 convolution, and an Identity. In the inference phase, the block is mathematically equivalently merged into a 3x3 convolution.

Since the RepVGG_like model needs to convert multiple parallel conv in the pretrained model into one conv when deploying the inference model, there are some question during quantization：
1. **Weight Outliers**: The equivalent conversion of the parameters leads to outliers in the weights of different channels on one conv, and it is difficult to quantify the weights; this phenomenon is especially obvious in the deployment model of MobileOne.
2. **Lack Of BN**: During the equivalent conversion of some RepVGG-like models during deployment, the conv and BatchNorm(BN) are fused into a new conv. There is no BN in quantization aware training(QAT), and the training is unstable and difficult to converge.

## Solution

To solve the above problems, we implement two tools to modify pretrained models to help quantization near-mathematically equivalently.
1. [BatchNorm Restoration(BNR)](../../../tinynn/util/bn_restore.py): BN restore tool. Since BN has been fused into the conv in the inference model of the RepVGG-like model, using the BNR tool to restore bn can effectively help quantization.
2. [Cross Layer Equalization(CLE)](../../../tinynn/graph/quantization/cross_layer_equalization.py): proposed by [Qualcomm](https://arxiv.org/abs/1906.04721). CLE is used to equalize the weights between different convolutions, which can well solve the problem of weight outliers in convolutions. After CLE, the quantization accuracy of the Re-parameter model can be greatly improved.
   1. After using CLE to homogenize the pre-trained model between different conv weights, a large part of the weight outliers can be eliminated, which greatly reduces the difficulty of weight quantization. CLE is mathematically equivalent.
   2. During the CLE operation, the Conv-BN will be fused, and the conv of the model will have bias. CLE will amplify some of the conv weights with small values,and at the same time cause the bias enlargement and activation enlargement, which make activation quantization difficult; In order to eliminate the impact of this phenomenon, High Bias Absorb(HBA) will be performed after CLE. This operation absorbs larger biases into subsequent conv.

## Usage

1. Post-training quantization: Refer to [`rep_ptq.py`](rep_ptq.py).
2. Quantization-aware training: Refer to [`rep_qat.py`](rep_qat.py).

## Experimental Results(CIFAR10)

| Model       | Top1 acc (%) | Top1 acc (%)<br/>(PTQ w/o CLE) | Top1 acc (%)<br/>(PTQ w/ CLE) |
|-------------|--------------|--------------------------------|-------------------------------|
| MobileOne   | 96.41        | 70.55(-25.86)                  | 95.44(-0.97)                  |
| RepVGG      | 94.46        | 46.35(-48.11)                  | 94.15(-0.31)                  |
| MobileNetV1 | 94.42        | 94.24(-0.18)                   | 94.30(-0.12)                  |

## Tips

1. The interface of BNR is `tinynn.util.bn_restore.model_restore_bn`, we give some parameter descriptions and usage examples below:
   1. `calibrate_func`: It needs to be given by the user to perform one complete epoch forward in train mode on the training dataset.
   2. `layers_fused_bn`: When performing BNR, the name list of convs which need BNR need to be explicitly given, default to be all convs in the given pretrained model.
   3. usage case:
```python
from tinynn.util.bn_restore import model_restore_bn

def calibare_func(model, context):
   model.train()
   for data in context.cali_loader:
      model(data)

device = torch.device('cpu') # Set according to the actual situation
context.cali_loader = train_dataloader # Set the corresponding calibarte data set using train dataset
# Specify a specific convs name list to perform BNR, otherwise a BN will be added after all convs by default.
layers_fused_bn = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d) and 'reparam' in name]
model_restore_bn(model, device, calibare_func, context, layers_fused_bn=layers_fused_bn)
```

2. HBA requires BN after conv. Since BN of the RepVGG-like deployment model has been merged into the conv, in order to apply HBA to improve the quantization accuracy, it is recommended to perform BNR first.
3. The interface of CLE is `tinynn.graph.quantization.cross_layer_equalization.cross_layer_equalize`, we give some parameter descriptions below:
   1. `threshold`: Invalid amplification prevention threshold. When CLE adjusts the weight, the maximum weight of some convs kernels is too small(less than 10e-23), and the bias and weight of the channel will be magnified by 10e10 times, resulting in the activation value also being magnified by 10e10 times, HBAs cannot remove this amplification. So we set `threshold` to prevent invalid amplification. Default to be 1000. During the CLE process, if a certain conv amplification is greater than 1000, the amplification will be rejected. If you find that some biases are significantly enlarged, you can reduce the threshold appropriately.
   2. `cle_iters`: The number of CLE iterations. When perform CLE once only, there may still be some outliers in the weights that cause quantization difficulties. Multiple CLEs can be performed to further improve the quantization accuracy. Default to be 2.
4. Optional operation: Before QAT, and after performing CLE and BNR on the deployment model of the RepVGG-like model, you can finetune training the floating-point model to make the model quantization training more stable.
