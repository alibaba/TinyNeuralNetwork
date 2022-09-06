# Identity pruning sample
[简体中文](README_zh-CN.md)

## Introduction
Identity pruning uses the variance of BatchNorm to determine the layers that have no effect on the network, and then eliminates this part of the redundancy.And achieve almost lossless accuracy.

## Identity pruning code description
The flow of the code is decribed as follows.
1. Create a `IdentityChannelPruner` object.
2. Call the `pruner.prune()` method to prune the provided model (The data and the dimensions of the weight tensor and other related tensors of the operators will be updated, thus you will get a smaller model.)
3. Perform finetuning on the pruned model.

## How do I adapt it to my model and code?
As for the dataset, you can see that the code comes with the utility functions for the cifar10 dataset. We also provide processing functions for datasets like cifar100 and imagenet (in the `tinynn.util` namespace). If you need to train with other datasets, you may need to write your own functions referring to our implementation.

As for the process, the training and validation functions need to be replaced by yours and the rest just remains.

As for the model, you just need to replace the model built by OneshotPruner object in step 2.

## Frequently Asked Questions

Because of the high complexity and frequent updates of PyTorch, we cannot ensure that all cases are covered through automated testing. When you encounter problems
You can check out the [FAQ](../../../docs/FAQ.md), or join the Q&A group in DingTalk via the QR Code below.

![img.png](../../../docs/qa.png)
