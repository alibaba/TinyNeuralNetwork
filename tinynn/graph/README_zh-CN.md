# 计算图捕获工具
[English](README.md)

## 背景

在不调用`torch.jit.trace`的情况下，我们很难获得计算图中节点的连接关系。然而，如果调用了该方法，有些节点会被优化掉，或是部分节点会被改写，我们很难将其与Python中的对象对应起来。因此我们设计了这样一套计算图的捕获机制。其中更奇妙的一点在于，借助于代码生成，可以将一个复杂的模型定义全部整理成一个代码文件。

## 样例使用
```py
import torch
import torchvision

from tinynn.graph.tracer import model_tracer, trace

with model_tracer():
    # 模型准备
    model = torchvision.models.alexnet()
    model.eval()

    # 提供一个可用的输入
    dummy_input = torch.rand((1, 3, 224, 224))

    # 通过trace，我们可以拿到计算图
    graph = trace(model, dummy_input)

    # 基于计算图，可以生成一份模型的描述代码
    graph.generate_code(f'my_alexnet.py', f'my_alexnet.pth', 'Alexnet')
```

## 量化
PyTorch的官方量化流程如下：
1. 修改模型的forward函数，插入`QuantStub、DeQuantStub`
2. 手动将模型中的所有`add、mul、cat`等操作替换为`torch.nn.quantized.FloatFunctional`，例如forward函数中的代码行`x = a + b`，需要创建一个对应的算子
  `self.add0 = torch.nn.quantized.FloatFunctional.add()` 然后在`forward`函数中调用 `x = self.add0.add(a, b)`
3. 编写`fuse_model`函数，手动进行算子的融合例如相邻的`Conv2D、BatchNorm2D`需要融合为`ConvBN2D`
  （越复杂的模型这一步操作也约复杂，尤其是一些网状结构，工作量惊人）
4. 调用`prepare_qat`将计算图转换为量化计算图（官方接口这一步是全图量化，对于混合精度量化需要自己魔改或者拆分模型才能支持）
5. 量化后的模型只能转为TorchScript，无法转为ONNX、TFLite等常见的端上结构导致难以部署

可以看到1-3步是很复杂的，也很容易出错，因此TinyNeuralNetwork中提供了自动的量化工具。

## 更多细节
1. 模型的实例化可以在with块中，也可以在with块外。我们更推荐在with块内，因为这样可以拿到模型构建时的原始参数。
2. 在一个with块中可以trace多个模型。
3. 我们支持在trace在推理时声明常量的模型，如果常量的shape较大，那么我们会把他转成Parameter来存放。

## 限制
1. 就像 `torch.jit.trace` 一样, 对于包含控制流OP的模型可能无法产生预期的输出。
2. 只会追踪PyTorch张量，其他的变量（比如Numpy张量、数值、字符串）都会变成常量。
3. 只会追踪部分张量的属性。打个比方，如果你调用张量的`.data` 或者 `.shape`属性，那么他会被加入到计算图里面。下面是tracer追踪的那些张量的属性。
    - `.data`
    - `.shape`
    - `.device`
    - `.dtype`
4. 不支持对于张量的`.size()` or `.shape`产生的`torch.Size`对象的调用`numel`方法，请使用`torch.prod`作为替代。
