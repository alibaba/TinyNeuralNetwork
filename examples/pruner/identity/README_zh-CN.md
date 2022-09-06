# IdentityPruner样例
[English](README.md)

## 简介
IdentityPruner通过BatchNorm的variance来判断对网络无影响的层，然后消除这部分冗余，并做到精度近乎无损。

## Oneshot剪枝代码说明
代码的流程如下
1. 创建IdentityChannelPruner对象
2. 调用 `pruner.prune()` 方法完成剪枝（修改算子的参数及权重维度）
3. 对剪枝后的模型做finetune

## 如何接入我的算法和模型？
从数据集角度而言，可以看到代码中自带了针对cifar10数据集的处理函数，我们也提供了cifar100以及imagenet的处理函数（位于tinynn.util命名空间下），如果没有你的数据集，可以参照着实现一个类似的。

从流程角度而言，除此之外的训练函数和验证函数，仍然可以使用之前的实现。

从模型角度而言，按照样例代码的流程，只需将第二步中构造OneshotPruner对象的模型进行替换即可。

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../../docs/qa.png)
