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

## 使用方法
可以支持对于模型直接进行转换，或者从`tinynn.util.converter_util.export_converter_files`生成json配置文件来做转换。

对于前者，可以参考`convert.py`，对于后者，可以参考`convert_from_json.py`。

## 常见问题

由于PyTorch具有极高的编码自由度，我们无法确保所有的Case都能自动化覆盖，当你遇到问题时，
可以查看[《常见问题解答》](../../docs/FAQ_zh-CN.md) ， 或者加入答疑群

![img.png](../../docs/qa.png)
