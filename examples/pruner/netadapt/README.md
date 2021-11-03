# NetAdapt pruning sample
[简体中文](README_zh-CN.md)

NetAdapt is a model pruning algorithm mentioned in the paper [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf).
TinyNeuralNetwork optimizes its generality and ease of use based on the implementation of [MIT](https://github.com/denru01/netadapt).

## NetAdapt pruning configuration
The parameters are as follows:

<dl>
<dt><tt>budget</tt>: int</dt>
<dd>The target value of the global resources. It is mutually exclusive with <tt>budget_ratio</tt></dd>
<dt><tt>budget_ratio</tt>: float</dt>
<dd>The target cropping ratio of the global resources. It is mutually exclusive with <tt>budget</tt></dd>
<dt><tt>budget_type</tt>: str</dt>
<dd>The target type of the global resource, valid values ​​include (flops)</dd>
<dt><tt>budget_reduce_rate_init</tt>: float</dt>
<dd>The initial value of the target cropping ratio of the resources in each round</dd>
<dt><tt>budget_reduce_rate_decay</tt>: float</dt>
<dd>The decay rate of the target cropping ratio of the resources in each round</dd>
<dt><tt>netadapt_lr</tt>: float</dt>
<dd>The learning rate for finetuning in NetAdapt</dd>
<dt><tt>netadapt_max_iter</tt>: int</dt>
<dd>The maximum iterations for finetuning in NetAdapt</dd>
<dt><tt>netadapt_max_rounds</tt>: int</dt>
<dd>(Optional) The maximum rounds for NetAdapt training. The default value is -1, which means no limit</dd>
<dt>netadapt_min_feature_size</tt>: int</dt>
<dd>(Optional) The minimum step length of the NetAdapt while reducing the number of channels. The default value is 8</dd>
</dl>

## NetAdapt pruning code description
The flow of the code is described as follows.
1. Parse the configuration file.
2. Generate the `NetAdaptPruner` object from the configuration file.
3. Generate a configuration file with pruning rates for each layer inplace. You can modify the pruning rates for specific operators. (optional)
4. Generate a `DLContext` object with parameters for NetAdapt training.
5. Call the `pruner.prune()` method to prune the provided model (The data and the dimensions of the weight tensor and other related tensors of the operators will be updated, thus you will get a smaller model.)
6. Select a suitable model from the generated models (considering the tradeoff between FLOPS and accuracy) and reload it with `pruner.restore(iteration)`.
7. Reload the selected model with `pruner.restore(iteration)`.
8. Generate a `DLContext` object with parameters for finetuning.
9. Perform finetuning on the reloaded model.

## How do I adapt it to my model and code?
As for the dataset, you can see that the code comes with the utility functions for the cifar10 dataset. We also provide processing functions for datasets like cifar100 and imagenet (in the `tinynn.util` namespace). If you need to train with other datasets, you may need to write your own functions referring to our implementation.

As for the process, the training and validation functions need to be replaced by yours and the rest just remains.

As for the model, you just need to replace the model built by NetAdaptPruner object in step 2.

## Frequently Asked Questions

Because of the high complexity and frequent updates of PyTorch, we cannot ensure that all cases are covered through automated testing. When you encounter problems
You can check out the [FAQ](../../../docs/FAQ.md), or join the Q&A group in DingTalk via the QR Code below.

![img.png](../../../docs/qa.png)
