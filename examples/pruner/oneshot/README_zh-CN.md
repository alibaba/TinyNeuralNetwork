# Oneshot剪枝样例
[English](README.md)

## 简介
OneShot剪枝是L1、L2、FPGM等结构化通道剪枝的统称（可以通过config文件中的metrics进行配置切换），是一种快速且通用性强的剪枝方案。

## Oneshot剪枝配置
参数如下：

<dl>
<dt><tt>sparsity</tt>  : float</dt>
<dd>稀疏率，所有算子都会减掉该比例的通道 </dd>
<dt><tt>metrics</tt>  : str</dt>
<dd>剪枝评估算法，有效值包括 (l1_norm, l2_norm, fpgm, random) </dd>
</dl>

当想调节特定算子的剪枝率时，建议通过 `pruner.generate_config(args.config)` 来生成一份包含所有可剪枝节点的配置（详情见demo）。

## Oneshot剪枝代码说明
代码的流程如下
1. 解析配置文件
2. 用配置文件生成OneshotPruner对象（解析配置、计算图、算子依赖）
3. 生成带每层剪枝率的配置文件，并修改特定算子的剪枝率（可选）
4. 调用 `pruner.prune()` 方法完成剪枝（修改算子的参数及权重维度）
5. 对剪枝后的模型做finetune

## 如何接入我的算法和模型？
从数据集角度而言，可以看到代码中自带了针对cifar10数据集的处理函数，我们也提供了cifar100以及imagenet的处理函数（位于tinynn.util命名空间下），如果没有你的数据集，可以参照着实现一个类似的。

从流程角度而言，除此之外的训练函数和验证函数，仍然可以使用之前的实现。

从模型角度而言，按照样例代码的流程，只需将第二步中构造OneshotPruner对象的模型进行替换即可。

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../../docs/qa.png)
