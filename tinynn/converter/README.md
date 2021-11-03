## Examples for model conversion
[简体中文](tinynn/converter/README_zh-CN.md)

## Background

Sometimes, we might need to convert PyTorch model to TFLite format in order to facilitate the deployment of the model on the device side. The existing conversion method usually takes the following procedures.
1. Convert to an ONNX model via `torch.onnx.export`
2. Convert to a Tensorflow frozen model via [onnx2tensorflow](https://github.com/onnx/onnx-tensorflow)
3. Convert to a TFLite model via `tensorflow.lite.TFLiteConverter`

This method has the following shortcomings.
1. The conversion is a lengthy process and often lead to problems
2. The conversion of quantized models is not supported
3. The models with LSTM cannot be converted
4. The model converted with onnx2tf has many redundant OPs

To solve the above problems, we implement this converter that translates models from PyTorch to TFLite directly.

## Features
1. Support for PyTorch 1.6+
2. Support for quantized models
3. Support for the LSTM op
4. A lot of optimization pass including continuous transpose, reshape elimination, no-op removal and etc.
5. Written in 100% Python, which is easy to maintain

## Code structure
+ [operators](tinynn/converter/operators): Most of the components of the converter
    + [tflite](tinynn/converter/operators/tflite) : TFLite related classes
        + [base.py](tinynn/converter/operators/tflite/base.py) : TFLite base data structure
        + [custom.py](tinynn/converter/operators/tflite/custom.py) : TFLite custom operators
        + [generated_ops.py](tinynn/converter/operators/tflite/generated_ops.py) : Wrapper class generated from TFLite schema
        + [transformable.py](tinynn/converter/operators/tflite/transformable.py) : Transformable operators, such as BatchNorm, Conv2d, and other composite operators composed of multiple TFLite operators
    + [torch](tinynn/converter/operators/torch) : PyTorch related classes
        + [base.py](tinynn/converter/operators/torch/base.py) : The base data structure needed for TorchScript parsing
        + [aten_schema.py](tinynn/converter/operators/torch/aten_schema.py) : Wrapper classes generated from ATen schema
        + [quantized_schema.py](tinynn/converter/operators/torch/quantized_schema.py) : Wrapper class generated from quantized schema
        + [aten.py](tinynn/converter/operators/torch/aten.py) : Translation of ATen-related operators
        + [quantized.py](tinynn/converter/operators/torch/quantized.py) : Translation of quantized-related operators
    + [base.py](tinynn/converter/operators/base.py) : Definition of generic operators
    + [graph.py](tinynn/converter/operators/graph.py) : Computation of graph-related infrastructure
    + [op_version.py](tinynn/converter/operators/op_version.py) : Handler for operator version
    + [optimize.py](tinynn/converter/operators/optimize.py) : Computation graph optimization
+ [base.py](tinynn/converter/base.py): Entry class `TFLiteConverter`
