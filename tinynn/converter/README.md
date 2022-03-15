## Examples for model conversion
[简体中文](README_zh-CN.md)

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
+ [operators](operators): Most of the components of the converter
    + [tflite](operators/tflite) : TFLite related classes
        + [base.py](operators/tflite/base.py) : TFLite base data structure
        + [custom.py](operators/tflite/custom.py) : TFLite custom operators
        + [generated_ops.py](operators/tflite/generated_ops.py) : Wrapper class generated from TFLite schema
        + [transformable.py](operators/tflite/transformable.py) : Transformable operators, such as BatchNorm, Conv2d, and other composite operators composed of multiple TFLite operators
    + [torch](operators/torch) : PyTorch related classes
        + [base.py](operators/torch/base.py) : The base data structure needed for TorchScript parsing
        + [aten.py](operators/torch/aten.py) : Translation of ATen-related operators
        + [quantized.py](operators/torch/quantized.py) : Translation of quantized-related operators
    + [base.py](operators/base.py) : Definition of generic operators
    + [graph.py](operators/graph.py) : Computation of graph-related infrastructure
    + [op_version.py](operators/op_version.py) : Handler for operator version
    + [optimize.py](operators/optimize.py) : Computation graph optimization
+ [schemas](schemas): Most of the schemas of the converter
    + [tflite](schemas/tflite) : TFLite related schemas
        + [schema_generated.py](schemas/tflite/schema_generated.py) : TFLite schema parsers
    + [torch](schemas/torch) : PyTorch related schemas
        + [aten_schema.py](schemas/torch/aten_schema.py) : Wrapper classes generated from ATen schema
        + [quantized_schema.py](schemas/torch/quantized_schema.py) : Wrapper class generated from quantized schema
        + [torchvision_schema.py](schemas/torch/torchvision_schema.py) : Wrapper class torchvision_schema from Torchvision schema
+ [base.py](base.py): Entry class `TFLiteConverter`
