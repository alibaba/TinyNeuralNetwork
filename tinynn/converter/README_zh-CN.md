# 模型转换样例
[English](README.md)

## 背景

为了方便模型在设备端的部署，有时我们需要将其转换为Tensorflow Lite的模型格式。
现有的转换方法主要会走以下的路径：
1. 通过torch.onnx.export转换为ONNX的模型
2. 通过onnx2tensorflow转换为tensorflow frozen model
3. 通过tensorflow的TFLiteConverter转换为TFLite的模型

这条路径存在以下的不足：
1. 转换路径较长，很容易产生问题
2. 无法支持量化模型的转换
3. 无法支持LSTM的模型
4. onnx2tf的模型存在很多冗余的OP

为了解决上述的问题，我们实现了从PyTorch到TFLite的直接转换器。

## 特性
1. 支持PyTorch 1.6+
2. 支持量化模型
3. 支持LSTM
4. 包含连续transpose和reshape消除、无用op删除等大量的优化pass
5. 纯Python编写，易于维护

## 代码结构
+ [operators](operators): 转换器的大部分组件
    + [tflite](operators/tflite) : TFLite相关的类
        + [base.py](operators/tflite/base.py) : TFLite基础数据结构
        + [custom.py](operators/tflite/custom.py) : TFLite自定义算子
        + [generated_ops.py](operators/tflite/generated_ops.py) : 从TFLite schema生成的Wrapper类
        + [transformable.py](operators/tflite/transformable.py) : 可转换算子，如BatchNorm、Conv2d等由多个TFLite算子组成的复合算子
    + [torch](operators/torch) : PyTorch相关的类
        + [base.py](operators/torch/base.py) : TorchScript解析所需的基础数据结构
        + [aten.py](operators/torch/aten.py) : ATen相关算子的翻译
        + [quantized.py](operators/torch/quantized.py) : Quantized相关算子的翻译
    + [base.py](operators/base.py) : 通用算子的定义
    + [graph.py](operators/graph.py) : 计算图相关的基础设施
    + [op_version.py](operators/op_version.py) : 设置算子版本
    + [optimize.py](operators/optimize.py) : 计算图优化
+ [schemas](schemas): schemas相关
    + [tflite](schemas/tflite) : TFLite相关的schema
        + [schema_generated.py](schemas/tflite/schema_generated.py) : TFLite schema 解析器
    + [torch](schemas/torch) : PyTorch相关的schema
        + [aten_schema.py](schemas/torch/aten_schema.py) : 从ATen schema生成的Wrapper类
        + [quantized_schema.py](schemas/torch/quantized_schema.py) : 从Quantized schema生成的Wrapper类
        + [torchvision_schema.py](schemas/torch/torchvision_schema.py) : 从Torchvision schema生成的Wrapper类
+ [base](base.py): 入口类TFLiteConverter
