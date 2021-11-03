# NetAdapt剪枝样例
[English](README.md)

NetAdapt 剪枝是论文 [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf) 中提到的一种模型剪枝算法。
TinyNeuralNetwork在 [MIT](https://github.com/denru01/netadapt) 实现的基础上优化了其通用性和易用性。

## NetAdapt剪枝配置
参数如下：

<dl>
<dt><tt>budget</tt>  : int</dt>
<dd>全局资源的目标值，与<tt>budget_ratio</tt>的设置互斥</dd>
<dt><tt>budget_ratio</tt>  : float</dt>
<dd>全局资源的目标裁剪比例，与<tt>budget</tt>的设置互斥</dd>
<dt><tt>budget_type</tt>  : str</dt>
<dd>全局资源的目标类型，有效值包括（flops）</dd>
<dt><tt>budget_reduce_rate_init</tt>  : float</dt>
<dd>每回合迭代资源的目标裁剪比例的初值</dd>
<dt><tt>budget_reduce_rate_decay</tt>  : float</dt>
<dd>每回合迭代资源的目标裁剪比例的衰减率</dd>
<dt><tt>netadapt_lr</tt>  : float</dt>
<dd>NetAdapt每回合finetune的学习率</dd>
<dt><tt>netadapt_max_iter</tt>  : int</dt>
<dd>NetAdapt每回合finetune的迭代数</dd>
<dt><tt>netadapt_max_rounds</tt>  : int</dt>
<dd>（可选）NetAdapt最大训练轮数，默认值为-1，表示不限制</dd>
<dt>netadapt_min_feature_size</tt>  : int</dt>
<dd>（可选）NetAdapt裁剪通道的最小步长，默认值为8</dd>
</dl>

## DLContext
### NetAdapt 训练
需要设定的参数如下：

<dl>
<dt><tt>train_loader</tt>  : Dataloader</dt>
<dd>训练用的dataloader</dd>
<dt><tt>val_loader</tt>  : Dataloader</dt>
<dd>训练用的dataloader</dd>
<dt><tt>criterion</tt>  : lambda (output: Tensor, target: Tensor) -> Tensor</dt>
<dd>损失函数</dd>
<dt><tt>optimizer</tt>  : Optimizer</dt>
<dd>优化器</dd>
</dl>

### Fine-tune
需要设定的参数如下：

<dl>
<dt><tt>train_loader</tt>  : Dataloader</dt>
<dd>训练用的dataloader</dd>
<dt><tt>val_loader</tt>  : Dataloader</dt>
<dd>训练用的dataloader</dd>
<dt><tt>criterion</tt>  : lambda (output: Tensor, target: Tensor) -> Tensor</dt>
<dd>损失函数</dd>
<dt><tt>optimizer</tt>  : Optimizer</dt>
<dd>优化器</dd>
<dt><tt>scheduler</tt>  : LR_Scheduler</dt>
<dd>（可选）优化器LR的调节器</dd>
</dl>

## NetAdapt剪枝代码说明
代码的流程如下
1. 解析配置文件
2. 用配置文件生成NetAdaptPruner对象
3. 生成带可选配置项的配置文件（可选）
4. 用NetAdapt训练的参数生成DLContext
5. 利用pruner的prune方法完成剪枝
6. 从生成的模型中选择一个合适的模型（考虑FLOPS和Acc的tradeoff）
7. 重新载入模型和pruner，使用pruner的restore方法恢复模型
8. 用finetune训练的参数生成DLContext
9. 对剪枝后的模型做finetune

## 如何接入我的算法和模型？
从数据集角度而言，可以看到代码中自带了针对cifar10数据集的处理函数，我们也提供了cifar100以及imagenet的处理函数（位于tinynn.util命名空间下），如果没有你的数据集，可以参照着实现一个类型的
从流程角度而言，除此之外的训练函数和验证函数，仍然可以使用之前的实现。
从模型角度而言，按照样例代码的流程，只需将第二步中构造NetAdaptPruner对象的模型进行替换即可。

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../../docs/qa.png)
