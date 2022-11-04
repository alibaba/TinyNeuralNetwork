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

## Usage
You may either perform conversion from models directly, or from the json configuration files generated using `tinynn.util.converter_util.export_converter_files`.

For the former one, you may refer to `convert.py`. And, for the latter one, you may refer to `convert_from_json.py`.

For dynamic quantization, you may refer to `dynamic.py`.

## Deployment options
a. NNAPI for CPU/GPU/NPU/XNNPACK (Android 8.1+, for quantized computational graphs, Android 10 and above is required)

b. For most devices, you can manually compile TFLite (C/C++) or use `tensorflow.lite.Interpreter` (Python)

c. For embedded devices, you can use TFLite Micro or other hardware-related inference engines

d. Generic inference engines with support of importing from the TFLite format (e.g MNN) can also be used

## Frequently Asked Questions

Because of the high complexity and frequent updates of PyTorch, we cannot ensure that all cases are covered through automated testing. When you encounter problems
You can check out the [FAQ](../../docs/FAQ.md), or join the Q&A group in DingTalk via the QR Code below.

![img.png](../../docs/qa.png)
