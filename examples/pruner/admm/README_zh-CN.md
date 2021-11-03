# ADMM剪枝样例
[English](README.md)

ADMM 剪枝是论文 [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294) 中提到的一种模型剪枝算法。

## ADMM剪枝配置
参数如下：

<dl>
<dt><tt>sparsity</tt>  : float</dt>
<dd>稀疏率，即每个OP需要剪枝的比率</dd>
<dt><tt>metrics</tt>  : str</dt>
<dd>剪枝评估算法，有效值包括 (l1, l2, fpgm, random) </dd>
<dt><tt>admm_iterations</tt>  : int</dt>
<dd>ADMM训练轮数，为总训练轮数除以<tt>admm_epoch</tt>的值 </dd>
<dt><tt>admm_epoch</tt>  : int</dt>
<dd>ADMM更新Z、U的周期 </dd>
<dt><tt>rho</tt>  : float</dt>
<dd>ADMM Loss中的惩罚因子 </dd>
<dt><tt>admm_lr</tt>  : float</dt>
<dd>ADMM训练过程的学习率 </dd>
</dl>

## ADMM剪枝代码说明
代码的流程如下
1. 解析配置文件
2. 用配置文件生成ADMMPruner对象
3. 生成带每层剪枝率的配置文件，并调节特定算子的剪枝率（可选）
4. 用admm稀疏化训练的参数生成DLContext
5. 利用pruner的prune方法完成剪枝
6. 用finetune训练的参数生成DLContext
7. 对剪枝后的模型做finetune

## 如何接入我的算法和模型？
从数据集角度而言，可以看到代码中自带了针对cifar10数据集的处理函数，我们也提供了cifar100以及imagenet的处理函数（位于tinynn.util命名空间下），如果没有你的数据集，可以参照着实现一个类型的
从流程角度而言，除此之外的训练函数和验证函数，仍然可以使用之前的实现。
从模型角度而言，按照样例代码的流程，只需将第二步中构造ADMMPruner对象的模型进行替换即可。

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../../docs/qa.png)

