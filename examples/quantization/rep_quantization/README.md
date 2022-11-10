# Rep_deploy model Quantization
[简体中文](README_zh-CN.md)

## Background

Rep-model(e.g. [RepVGG](https://arxiv.org/abs/2101.03697), [MobileOne](https://arxiv.org/abs/2206.04040)): Such models construct a series of multi-branch structures during training and merge the parameters into a single-branch structure during the inference.
For example, a block of the RepVGG in the training phase is a parallel branch structure consisting of a 3x3 convolution, a 1x1 convolution, and an Identity. In the inference phase, the block is mathematically equivalently merged into a 3x3 convolution.

Since the parameters of the pre-trained model need to be merged when deploying the inference model, there are some question during quantization：
1. During the parameter merging process converted to the deployment model, the BatchNorm operator has been merged into the convolution. There is no BN layer during Quantization-aware training, so that the training is unstable and even cannot be converged.
2. Due to the merging of parameters, the convolution kernels of different channels, especially for DW convolution (in MobileOne), have some outliers in their weights (in our experiments, we found that the maximum absolute value of the weights of partial convolutions is greater than 100). When using per_tensor MinMax quantization of weights, the accuracy drops a lot.

## Solution

To solve the above problems, we implement two tools to modify pretrained models mathematically equivalently to help quantization.
1. [BN_Restore](../../../tinynn/util/bn_restore.py): Since BN has been fused into the convolution in the inference model of the Re-parameter model, using the `BN_Restore` tool to restore bn can effectively improve the convergence and stability of Quantization-aware training (Post-training quantization does not require BN restore).
2. Cross_Layer_Equalization ([CLE](../../../tinynn/graph/quantization/cross_layer_equalization.py)): proposed by [Qualcomm](https://arxiv.org/abs/1906.04721). CLE is used to equalize the weights between different convolutions, which can well solve the problem of weight outliers in convolutions. By using CLE, the quantization accuracy of the Re-parameter model can be greatly improved.

## Usage

1. Post-training quantization: Refer to `rep_ptq.py`.
2. Quantization-aware training: Refer to `rep_qat.py`.

## Tips

1. The interface of BN_Restore is `tinynn.util.bn_restore.model_restore_bn`, which needs to provide a `calibrate_func` that performs calibration in train mode on the full training set.
In addition, the name list of convolution which need bn_restored is specified by the parameter `layers_fused_bn`, defaults to all convolution in the given model.
2. Cross_Layer_Equalization is an optional operation whose interface is `tinynn.graph.quantization.cross_layer_equalization.cross_layer_equalize`.
In our experiments, CLE can well solve the problem of weight outliers to effectively improve the accuracy of post-quantization and ensure the initial accuracy of Quantization-aware training.
3. CLE operation needs to be done before BN_restore operation.
4. When using CLE, we found that the feature output of the partial convolution will be enlarged by 10e8 times, resulting in the failure of activation quantization. You can set the parameter `threshold` to avoid activation quantization failure.
In addition, when CLE is performed once alone, the degree of uniformity of the weights is sometimes not guaranteed. You can try to perform CLE multiple times to further improve the quantization accuracy.
5. Optional operation: When preparing to Quantization-aware training, after performing CLE and BN_restore on the rep—deploy-model, you can finetune the model to make Quantization-aware training more stable.
