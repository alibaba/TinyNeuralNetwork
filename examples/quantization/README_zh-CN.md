# 量化
[English](README.md)

TinyNeuralNetwork采用PyTorch的量化（[PyTorch 量化教程](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) ）
模块作为后端，但着重优化了其易用性（ **尽量使用PyTorch 1.6及以后的版本，量化这块迭代很快** ）。

## PyTorch量化流程
1. 修改模型的forward函数，插入`QuantStub、DeQuantStub`
2. 手动将模型中的所有`add、mul、cat`等操作替换为`torch.nn.quantized.FloatFunctional`，例如forward函数中的代码行`x = a + b`，需要创建一个对应的算子
  `self.add0 = torch.nn.quantized.FloatFunctional()` 然后在`forward`函数中调用 `x = self.add0.add(a, b)`
3. 编写`fuse_model`函数，手动进行算子的融合例如相邻的`Conv2D、BatchNorm2D`需要融合为`ConvBN2D`
  （越复杂的模型这一步操作也越复杂，尤其是一些网状结构，工作量惊人）
4. 调用`prepare_qat`将计算图转换为量化计算图（官方接口这一步是全图量化，对于混合精度量化需要自己魔改或者拆分模型才能支持）
5. 量化后的模型只能转为TorchScript，无法转为ONNX、TFLite等常见的端上结构导致难以部署


## TinyNeuralNetwork量化流程
+ 基于用户提供的模型代码，codegen一份新的代码（*等价于上述的步骤1，步骤2*）
+ 提供更智能的`prepare_qat`函数，内部自动完成计算图的算子融合，以及混合精度的分析（*等价于上述步骤3， 4*）
+ 提供TorchScript到TFLite的转换（*支持浮点、量化模型*），简化端上部署

## 支持的量化方法
- [训练时量化](qat.py)
- [训练后量化](post.py)
- 动态量化 [使用 TFLiteConverter (推荐)](../converter/dynamic.py) / [使用 DynamicQuantizer (不推荐)](dynamic.py)
- [BFloat16量化](bf16.py)

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../docs/qa.png)
