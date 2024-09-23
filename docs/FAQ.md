# FAQ
[简体中文](FAQ_zh-CN.md)

## Pruning
#### How do I modify the pruning rate of a single operator?
Q: In the example, we set the pruning rate of the whole network. How to adjust the pruning rate of a specific layer?
```yaml
# example config
sparsity: 0.25
metrics: l2_norm # The available metrics are listed in `tinynn/graph/modifier.py`
```

A: After calling `pruner.prune()`, a new configuration file with the sparsity for each operator will be generated inplace. You can use this file as the configuration for the pruner or generate a new configuration file based on this one. (e.g. line 42 in `examples/oneshot/oneshot_prune.py`)

```yaml
# new yaml generated
sparsity:
  default: 0.25
  model_0_0: 0.25
  model_1_3: 0.25
  model_2_3: 0.25
  model_3_3: 0.25
  model_4_3: 0.25
  model_5_3: 0.25
  model_6_3: 0.25
  model_7_3: 0.25
  model_8_3: 0.25
  model_9_3: 0.25
  model_10_3: 0.25
  model_11_3: 0.25
  model_12_3: 0.25
  model_13_3: 0.25
metrics: l2_norm # Other supported values: random, l1_norm, l2_norm, fpgm
```

#### How to speed up training?
The training in TinyNeuralNetwork is based on PyTorch. Usually, the bottleneck is in the data processing part, which you can try to use LMDB and other in-memory databases to accelerate.

## Quantization

#### How to deal with errors related to quantization?
Q: Some operators such as max_pool2d_with_indices will fail when quantizing

A: The quantization-aware training of TinyNeuralNetwork is based on that of PyTorch, and only reduces its complexity related to operator fusion and computational graph rewrite.
TinyNeuralNetwork does not support operators that are not natively supported by PyTorch, such as LeakyReLU and etc. The full table may be seen [here](quantization_support.md#unsupported-operators-in-pytorch-for-static-quantization). Please wrap up `torch.quantization.QuantWrapper` on those modules.
(*More operators are supported in higher versions of PyTorch. So, please consult us first or try a higher version if you encounter any failure*)

#### How to perform mixed precision quantization?
Q: How to quantize only part of a quantized graph when the default is to perform quantization on the whole graph?

```python
# Quantization with the whole graph
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out')
    qat_model = quantizer.quantize()
```

A: First, perform quantization for the whole graph. Then, the easy way is to use the layerwise configuration file. Please refer to the code example [here](../examples/quantization/selective_q.py). If that doesn't work, you may modify the positions of QuantStub and DeQuantStub manually. After that, using the code below to load the model. The detailed code example can be seen [here](../examples/mixed_qat/qat.py).
```python
# Reload the model with modification
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'force_overwrite': False})
    qat_model = quantizer.quantize()
```

Q: How to specify mixed quantization according to operator types?

A: Configure the quantize_op_action parameter in the config during Quantizer initialization. You need to specify the actions for non-quantized operators: 'disable' means completely non-quantized, and 'rewrite' means not quantized but retaining the quantization parameters of the operator's inputs and outputs.

```python
# For a model containing LSTM op, perform mixed quantization while retaining the quantization parameters of its inputs, facilitating subsequent quantization directly in the converter.
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={ 'quantize_op_action': {nn.LSTM: 'rewrite'} })
    qat_model = quantizer.quantize()
```


#### How to set a more flexible Qconfig?
Q: How to set different quantization configurations, such as specifying different quantization observers for different layers?

A: Configure the `override_qconfig_func` parameter in the config during `Quantizer` initialization. This requires the user to define a function that modifies the Qconfig for the corresponding Op. Below is an example to set MinMaxObservers based on different module name or module type. More `FakeQuantize` and `Observer` implementations can be selected from the official `torch.quantization` library, or you can [customize your own implementations](../tinynn/graph/quantization/fake_quantize.py).

module_name can be obtained from the generated traced model definition in out/Qxx.py.

```python
import torch
from torch.quantization import FakeQuantize, MinMaxObserver
form torch.ao.nn.intrinsic import ConvBnReLU2d
def set_MinMaxObserver(name, module):
   # Set the corresponding weight and activation observers to MinMaxObserver based on model_name and module_type.
   if name in ['model_0_0', 'model_0_1'] or isinstance(module, ConvBnReLU2d):
        weight_fq = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        )
        act_fq = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            reduce_range=False,
        )
        qconfig_new = torch.quantization.QConfig(act_fq, weight_fq)
        return qconfig_new
```
```python
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'override_qconfig_func': set_MinMaxObserver})
    qat_model = quantizer.quantize()
```


#### How to handle the case of inconsistent training and inference computation graphs?

Q: Models may have some extra logic in the training phase that are not needed in inference, such as the model below (which is also a common scenario in real world OCR and face recognition).
This will result in the quantization model code generated by codegen during training is not available for inference.

```python
class FloatModel(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d()
        self.conv1 = nn.Conv2d()

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            x = self.conv1(x)
        return x

```

A: There are generally two ways to tackle this problem.
+ Use the code generator in TinyNeuralNetwork to create `qat_train_model.py`, `qat_eval_model.py` in case of `model.train()`, `model.eval()`, respectively
  Use `qat_train_model.py` for training, and then use `qat_eval_model.py` to load the weights trained by the former when inference is needed
  (Since there is no `self.conv1` in `qat_eval_model.py`, you need to set `strict=False` when calling `load_state_dict`)
+ Like the former one, generate two different copies of the model in training mode and evaluation mode respectively. And then, make a copy of `qat_train_model.py` and replace the forward function with that in `qat_eval_model.py` manually. Finally, use the modified script as the one for the evaluation mode.

#### How to fuse normalization in preprocessing and the Quantize OP, so that the raw image data is used as input?

Assuming normalization is done in preprocessing using `normalized = (image - mean) / std`, you can pass in the parameter `'quantized_input_stats': [(mean, std)]` when constructing `Quantizer`, as well as constructing `Converter` with `fuse_quant_dequant=True`, then the image data (`image` in the formula) can be passed in as the `uint8` data format.

For example, the following preprocessing process is often used for images in torchvision.

```py
transforms = transforms.Compose(
    [
        Resize(img_size),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
```

Except for the Resize process, `ToTensor` converts the data to floating point and divides it by 255, then `Normalize` performs normalization according to `mean=(0.4914, 0.4822, 0.4465)` and `std=(0.2023, 0.1994, 0.2010)`. In this case, it is simliar to the normalization of `mean=114.3884` and `std=58.3021`, which of course leads to some accuracy loss. If you want to have higher accuracy, you can try out the following things.

1. Unify the normalization parameters of the channels before training the floating-point model or before QAT training
2. Make sure `mean` is set to an integer, because the corresponding quantization parameter `zero_point` can only be an integer.

P.S. For inputs of the `int8` type , you may need to perform the `uint8` to `int8` conversion yourself before feeding to the model as input (subtract 128 manually)

#### The accuracy of PyTorch's post-quantization algorithm is seriously degraded. Are there other post-quantization algorithms?
At present, PyTorch officially uses L2_norm as the post-quantization algorithm, and TinyNeuralNetwork supports the KL divergence-based post-quantization algorithm on the basis of PyTorch. When instantiating `PostQuantizer`, set the corresponding `algorithm` in config. The default algorithm option is `l2`, and the optional options are `l2`, `kl`.
```py
 with model_tracer():
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    dummy_input = torch.rand((1, 3, 224, 224))

    # Set the algorithm options you need, the default option is l2.
    quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'algorithm':'kl'})
    ptq_model = quantizer.quantize()
```

#### How to restore a quantization-rewritten model to the original floating point model?
We have designed `DeQuantizer` for this purpose. Please refer to the code below.
```py
# `model_qat` is the generated model definition of the TinyNN graph tracer,
#  which is typically in the "out/" directory.
model = model_qat()
dummy_input = torch.randn(1, 3, 224, 224)
dequantizer = DeQuantizer(model, dummy_input, work_dir='out')
float_model = dequantizer.dequantize()
```

## Model conversion

#### What should I do if the operator is not supported?
There are a large number of operators in PyTorch. We cannot cover all operators, but only most of the commonly used ones. Therefore, if you have unsupported operators in your model, you have the following options:
1. [Submit](https://github.com/alibaba/TinyNeuralNetwork/issues/new/choose) a new issue
2. You can also try to implement it yourself. The process of operator translation in model conversion is actually the process of mapping between corresponding TorchScript OP and TFLite OP.

    The locations of the relevant code
    - TFLite
        - OP schema (without I/O tensors): [generated_ops.py](../tinynn/converter/operators/tflite/generated_ops.py)
        - Full schema: https://www.tensorflow.org/mlir/tfl_ops
    - TorchScript
        - ATen schema [aten_schema.py](../tinynn/converter/schemas/torch/aten_schema.py)
        - Quantized schema [quantized_schema.py](../tinynn/converter/schemas/torch/quantized_schema.py)
        - Quantized schema [torchvision_schema.py](../tinynn/converter/schemas/torch/torchvision_schema.py)
    - Translation logic
        - ATen OPs [aten.py](../tinynn/converter/operators/torch/aten.py)
        - Quantized OPs [quantized.py](../tinynn/converter/operators/torch/quantized.py)
    - Registration of OP translation logic
        - Registration table [\_\_init\_\_.py](../tinynn/converter/operators/torch/__init__.py)

    Implementation steps:
    1. Read through the schema of both TorchScript and TFLite, and select the appropriate OP(s) on both sides
    2. Add an entry in the OP translation registration table
    3. Add a new parser class to the translation logic. This class needs to inherit the corresponding TorchScript schema class.
    4. Implement the function `parse` of the aforementioned class

    For details, please refer to the implementation of SiLU: https://github.com/alibaba/TinyNeuralNetwork/commit/ebd30325761a103c5469cf6dd4be730d93725356

#### Model conversion fails for unknown reasons. How to provide the model to the developers for debugging purposes?
You can use `export_converter_files` to export your models with some related configuration files. For details, see the code below
````py
from tinynn.util.converter_util import export_converter_files

model = Model()
model.cpu()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
export_dir = 'out'
export_name = 'test_model'

export_converter_files(model, dummy_input, export_dir, export_name)
````
Executing this code, you'll get two files in the specified directory, including the TorchScript model (.pt) and the input and output description files (.json). These two files can be shared with developers for debugging.

#### Why is the input/output tensor shape different from the one in the original model?
Generally, for a vision model, the memory layout of the input data used by PyTorch is NCHW, and on the embedded device side, the layout of the supported image data is usually NHWC. Therefore, the 4-dimensional input and output is transformed by default. If you do not need this behaviour, you can add the parameter `nchw_transpose=False` (or `input_transpose=False` and `output_transpose=False`) when defining TFLiteConverter.

#### Why the converted model with grouped (de)convolution does not work?
Since TFLite does not officially support grouped (de)convolution, we have extended the implementation of grouped (de)convolution internally based on the `CONV_2D` and the `TRANSPOSE_CONV` operator. To generate a standard TFLite model, you can add the parameter `group_conv_rewrite=True` when defining TFLiteConverter.

#### What if `BidirectionalLSTM` is unsupported while `UnidirectionalLSTM` is supported?
You may add the parameter `map_bilstm_to_lstm=True` when defining TFLiteConverter.

#### How to convert a model with LSTM or GRU?
The easy way is to pass in `unroll_rnn=True` when defining TFLiteConverter, so that everything works just like in PyTorch. But the `LSTM/GRU`s will be translated to a bunch of ops, which makes the computation graph complicated. Alternatively, if you need a single op for each `LSTM`(GRU is not supported yet), then you may refer to the content below.

Since the target format of our conversion is TFLite, we need to understand how LSTM works in PyTorch and Tensorflow respectively.

When using TF2.X to export the LSTM model to Tensorflow Lite, it will be translated into the `UnidirectionalLSTM` operator, and the state tensors in it will be saved as a `Variable` (a.k.a persistent memory). The state of each mini-batch will be automatically be accumulated. These state tensors are not included in the input and output of the model.

In PyTorch, LSTM contains an optional state input and state output. When the state is not passed in, the initial hidden layer state always remains all 0 for each mini-batch inference, which is different from Tensorflow.

Therefore, in order to simulate the behavior on the Tensorflow side, when exporting the LSTM model on the PyTorch side, be sure to delete the LSTM state inputs and outputs from the model inputs and outputs.

Next, for streaming and non-streaming scenarios, how should we use the exported LSTM model?

##### Non-streaming scenarios
In this case, we just need to set the state inputs to 0. Fortunately, Tensorflow Lite's Interpreter provides a convenient interface [reset_all_variables](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter#reset_all_variables).
So, we only need to call `reset_all_variables` before each call to `invoke`.

##### Streaming scenarios
In this case, it's somehow more complicated because we need to read and write state variables. We can use Netron to open the generated model, locate all LSTM nodes, and view the input whose name contains state. For example, for the states in a unidirectional LSTM node, the attributes are named `output_state_in` and `cell_state_in`, you can expand and see that their kind is `Variable`. Record their locations (i.e. the `location` property). You may use `tinynn.converter.utils.tflite.parse_lstm_states(tflite_path)` for collecting the indices of the state tensors in the TFLite model.

![image](https://user-images.githubusercontent.com/9998726/150492819-5f7d4f43-347e-4c1f-a700-72e77d95e9e9.png)

When using Tensorflow Lite's Interpreter, you only need to read or write these state variables according to these `location`s, combined with methods like `get_tensor` and `set_tensor`. For details, see [here](https://github.com/alibaba/TinyNeuralNetwork/issues/29#issuecomment-1018292677).

Note: These state variables are all two-dimensional with the shape of `[batch_size, hidden_size or input_size]`. So in the streaming scenario, you only need to split these variables according to the first dimension.

#### How to speed up inference for LSTMs?
Usually, when the number of hidden layers is large enough (128+), the LSTM OP will be time-consuming in the TFLite backend. In this case, consider using dynamic range quantization to optimize its performance, see [dynamic.py](../examples/converter/dynamic.py).

You may also try out static quantization for LSTMs when you have PyTorch 1.13+. But it may take much more effort to minimize the quantization error, and you probably need to perform per-layer inspection carefully.
We also support int16 LSTM via the combination of static quantization and LSTM-only dynamic quantization. Please take a look at [ptq_with_dynamic_q_lstm.py](../examples/quantization/ptq_with_dynamic_q_lstm.py).

#### What if my model runs slower when dynamic quantization is enabled?
Please refer to [dynamic_with_selection.py](../examples/converter/dynamic_with_selection.py) for selective dynamic quantization.

#### I need LSTM/GRUs with separated gate calculation when `unroll_rnn=True`.
Please set `separated_rnn_gate_calc=True`.

#### How to add state inputs and outputs for LSTMs/GRUs/RNNs with `unroll_rnn=True`?
It is possible to rewrite the model using the Graph Tracer and Code Generator of TinyNN. Please use the following code.
```py
from tinynn.graph.tracer import trace
graph = trace(model, dummy_input)
graph.add_state_input_outputs()
graph.inplace_commit(True)
```

P.S. Avoid using `rnn.flatten_parameters()`. Otherwise, `torch.jit.trace` may fail。

#### What if duplicate tensors is generated in the TFLite model (e.g. when performing static quantization for LSTMs)?
You may try out `group_tensors=True` to remove those duplicates.

## Quantized model conversion

##### How to convert ops that cannot be quantized in PyTorch to quantized kernels, e.g. `SOFTMAX`, `LOG_SOFTMAX` and `BATCH_MATMUL`?
You may refer to this [table](quantization_support.md#extra-flags-for-translating-the-above-ops-to-quantized-tflite).

## Interoperability with other frameworks

### HuggingFace Transformers
Some of the models in [huggingface/transformer](https://github.com/huggingface/transformers) including `ViTForImageClassification` preloads the PyTorch functions during module import, which breaks our logic for tracing. You may need to use the `import_patcher` for model pruning and quantization.

```py
# Import import_patcher from TinyNN
from tinynn.graph.tracer import import_patcher

# Apply import_patcher during module import for transformers
with import_patcher():
    from transformers import ViTForImageClassification
```

### ONNX2PyTorch
[ToriML/onnx2pytorch](https://github.com/ToriML/onnx2pytorch) is a project that translates ONNX models to PyTorch, so that TinyNN can be used to perform model compression. For pruning and quantization to work, you may follow the logic below.
```py
# Import import_patcher from TinyNN
from tinynn.graph.tracer import import_patcher
# Import ConvertModel from onnx2pytorch
from onnx2pytorch import ConvertModel

# Apply import_patcher during module conversion for onnx2pytorch
    with import_patcher():
        model = ConvertModel(onnx_model)
```

### ONNX2Torch
[ENOT-AutoDL/onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch) is a new project that converts AI models from ONNX to PyTorch. As for handling dynamic shape, we need to use some hacks while using TinyNN for pruning and quantization.
```py
# Import import_patcher from TinyNN
from tinynn.graph.tracer import import_patcher
# Apply import_patcher during module import for onnx2torch
with import_patcher():
    from onnx2torch import convert

model = convert(onnx_model)

# Graph tracing
graph = trace(model, dummy_input, patch_torch_size=True)
graph.generate_code('my_model.py', 'my_model.pth', 'MyModel')

# Quantization
quantizer = PostQuantizer(model, dummy_input, config={'extra_tracer_opts': {'patch_torch_size': True}})
ptq_model = quantizer.quantize()
```
