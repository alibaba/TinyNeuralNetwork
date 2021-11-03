# Oneshot pruning sample
[简体中文](README_zh-CN.md)

## Introduction
OneShot pruning is the collective name for structured channel pruning of L1, L2, FPGM, etc. (which can be switched by configuration via metrics in the config file), and is a fast and versatile pruning solution.

## Oneshot pruning configuration
The parameters are listed as follows.

<dl>
<dt><tt>sparsity</tt> : float</dt>
<dd>Sparsity: all prunable operators remove channels with this ratio </dd>
<dt><tt>metrics</tt> : str</dt>
<dd>Evaluation method used for channel pruning, valid values include (l1_norm, l2_norm, fpgm, random) </dd>
</dl>

If you want to adjust the pruning rate of a particular operator, we will recommend that generating a configuration file with all prunable nodes via `pruner.generate_config(args.config)` (see the demo for details).

## Oneshot pruning code description
The flow of the code is decribed as follows.
1. Parse the configuration file
2. Generate a `OneshotPruner` object from the configuration file (parsing configuration and dealing with dependencies between operators in the computation graph)
3. Generate a configuration file with pruning rates for each layer inplace. You can modify the pruning rates for specific operators. (optional)
4. Call the `pruner.prune()` method to prune the provided model (The data and the dimensions of the weight tensor and other related tensors of the operators will be updated, thus you will get a smaller model.)
5. Perform finetuning on the pruned model.

## How do I adapt it to my model and code?
As for the dataset, you can see that the code comes with the utility functions for the cifar10 dataset. We also provide processing functions for datasets like cifar100 and imagenet (in the `tinynn.util` namespace). If you need to train with other datasets, you may need to write your own functions referring to our implementation.

As for the process, the training and validation functions need to be replaced by yours and the rest just remains.

As for the model, you just need to replace the model built by OneshotPruner object in step 2.

## Frequently Asked Questions

Because of the high complexity and frequent updates of PyTorch, we cannot ensure that all cases are covered through automated testing. When you encounter problems
You can check out the [FAQ](../../../docs/FAQ.md), or join the Q&A group in DingTalk via the QR Code below.

![img.png](../../../docs/qa.png)

