# FAQ
[English](FAQ.md)

## 剪枝
#### 如何修改单个OP的剪枝率？
Q：在example中是设置的整个网络的剪枝率，如何调整某个特定OP的剪枝率？
```yaml
# example中的yaml
sparsity: 0.25
metrics: l2_norm # The available metrics are listed in `tinynn/graph/modifier.py`
```

A：prune在解析了剪枝率后，会生成每个OP对应的稀疏率， 你可以直接修改也可以生成一份新的yaml
（例如oneshot的example中的第42行）
```yaml
# 生成的新的yaml
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
metrics: l2_norm # 除此之外，还可以使用 random, l1_norm, l2_norm, fpgm
```

#### 训练速度太慢如何解决？
TinyNeuralNetwork的训练依托于PyTorch，通常瓶颈都是在数据处理部分，可以尝试使用LMDB等技术来进行数据读取的加速

## 量化

#### 算子量化失败如何处理？
Q：有的算子例如max_pool2d_with_indices在量化的时候会失败

A：TinyNeuralNetwork的量化训练是使用PyTorch的量化训练作为后端，仅优化了其算子融合与计算图转换相关的逻辑。PyTorch原生
不支持的算子TinyNeuralNetwork也无法支持例如ConvTrans2D、max_pool2d_with_indices、LeakyReLU等等（*高版本的PyTorch
支持的算子更多， 遇到失败的情况可以第一时间咨询我们或者尝试更高的版本*)

#### 如何实现混合精度量化？
Q： 量化计算图生成默认是全图量化，如何只量化其中一部分？
```python
# 全图量化
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out')
    qat_model = quantizer.quantize()
```

A：先进行全图量化，然后手工修改QuantStub、DeQuantStub的位置，之后使用下面的代码来加载模型。
```python
# 载入修改后的模型代码
with model_tracer():
    quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'force_overwrite': False})
    qat_model = quantizer.quantize()
```


#### 如何处理训练和推理计算图不一致的情况？

Q：许多模型在训练阶段会运行一些额外的算子，而在推理时不需要，例如下述模型（真实情况下OCR、人脸识别也常遇到此种场景）。
这会导致在训练时通过codegen生成的量化模型代码是无法用于推理的。

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

A：一般有两种解法
+ 在model.train()，model.eval()情况下分别codegen得到qat_train_model.py, qat_eval_model.py，
  用前者进行训练，然后在需要推理的时候用qat_eval_model.py去load前者训练出来的权重
  （由于qat_eval_model.py中并没self.conv1，因此load_state_dict的时候需要设置strict=False)
+ 仍然生成两份代码，然后复制一份qat_train_model.py并把forward函数手动替换为qat_eval_model.py中的forward函数即可

#### 如何将预处理中的Normalization和Quantize OP融合起来，将图像原始数据作为输入？

假设预处理中使用 `normalized = (image - mean) / std` 来做 normalization，可以在构造Quantizer的时候传入参数 `'quantized_input_stats': [(mean, std)]`，以及在Converter构造时传入`fuse_quant_dequant=True`，然后就可以将图片数据（公式中的`image`）以`uint8`的数据格式传入。

举例来说，对于torchvision中的图像常采用如下的预处理流程。

```py
transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
```

除去Resize过程，`ToTensor` 会将数据转变成浮点型，然后除以255，然后 `Normalize` 按照 `mean=(0.4914, 0.4822, 0.4465)` 以及 `std=(0.2023, 0.1994, 0.2010)` 做Normalization。在这种情况下，我们可以将其等效看作完成了 `mean=114.3884` 和 `std=58.3021` 的 Normalization。当然这种情况下，会导致一些精度损失。如果想有更高的精度，可以尝试

1. 在训练浮点模型前或者QAT训练前尽量将通道的Normalization参数统一
2. 尽量确保 `mean` 设置的是一个整数，因为对应的量化参数 `zero_point` 只能是一个整数。

P.S. 对于 `int8` 类型的输入，你可能需要在模型输入前自行完成 `uint8` 到 `int8` 的转换 （手工减128）

#### PyTorch的后量化算法精度下降比较严重，有没有其他的后量化算法？

目前PyTorch官方使用L2 norm作为后量化算法，TinyNeuralNetwork在PyTorch的基础上支持了基于KL散度的后量化算法。在实例化`PostQuantizer`时，在config中设置对应的`algorithm`，目前默认的algorithm选项为`l2`，可选的选项为`l2`、`kl`。
```py
 with model_tracer():
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    dummy_input = torch.rand((1, 3, 224, 224))

    # 设置你需要的algorithm选项，默认为l2。
    quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'algorithm':'kl'})
    ptq_model = quantizer.quantize()
```


## 模型转换

#### 算子不支持如何处理？
由于PyTorch的算子数量相当多，我们无法覆盖所有的算子，只能覆盖大部分常用的算子。因此，如果遇到不支持的算子，你可以有以下选择：
1. [提交](https://github.com/alibaba/TinyNeuralNetwork/issues/new/choose)一个新issue
2. 你也可以选择自己实现，模型转换中算子翻译的过程其实就是将TorchScript OP与TFLite OP对应的过程

    相关代码的位置
    - TFLite
        - OP schema (without I/O tensors): [generated_ops.py](../tinynn/converter/operators/tflite/generated_ops.py)
        - Full schema: https://www.tensorflow.org/mlir/tfl_ops
    - TorchScript
        - ATen schema [aten_schema.py](../tinynn/converter/schemas/torch/aten_schema.py)
        - Quantized schema [quantized_schema.py](../tinynn/converter/schemas/torch/quantized_schema.py)
        - Torchvision schema [torchvision_schema.py](../tinynn/converter/schemas/torch/torchvision_schema.py)
    - 两者的对应翻译代码
        - ATen OPs [aten.py](../tinynn/converter/operators/torch/aten.py)
        - Quantized OPs [quantized.py](../tinynn/converter/operators/torch/quantized.py)
    - OP翻译逻辑的注册
        - Registration [\_\_init\_\_.py](../tinynn/converter/operators/torch/__init__.py)

    实现步骤：
    1. 查阅TorchScript和TFLite的Schema，选取两边对应的OP
    2. 在OP翻译注册逻辑中添加一个条目
    3. 在翻译对应代码处添加对应的类，该类需继承相应的TorchScript schema类。
    4. 在上述类中添加对应逻辑

    具体可以参见SiLU的实现： https://github.com/alibaba/TinyNeuralNetwork/commit/ebd30325761a103c5469cf6dd4be730d93725356

#### 模型转换因为未知原因失败了，如何提供相应的数据方便开发者调试？
我们提供了一个函数，方便进行模型和相关配置的导出，具体可见下方代码
```py
from tinynn.util.converter_util import export_converter_files

model = Model()
model.cpu()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
export_dir = 'out'
export_name = 'test_model'

export_converter_files(model, dummy_input, export_dir, export_name)
```
执行这段代码会在指定的目录下生成两个文件，包含TorchScript模型(.pt)和输入输出的描述文件(.json)，可以将这两个文件分享给开发者来做调试。

#### 为什么输入输出的shape和原始的不一样？
一般在视觉模型中，PyTorch这边采用的输入数据的内存排布为NCHW，而在嵌入式设备侧，一般支持的图片数据的排布为NHWC。因此，默认对4维的输入输出都做了内存排布的转换，如果你不需要这种转换，可以在定义TFLiteConverter时加上`nchw_transpose=False`这个参数(或是`input_transpose=False`以及 `output_transpose=True`)。

#### 为何有分组(反)卷积的模型转换出来无法运行？
由于TFLite官方无分组(反)卷积的支持，我们在内部基于`CONV_2D`和`TRANSPOSE_CONV`算子拓展了分组(反)卷积的实现。如需生成标准的TFLite模型，可以在定义TFLiteConverter时加上`group_conv_rewrite=True`这个参数。

#### 如果我的部署平台只支持`UnidirectionalLSTM`，不支持`BidirectionalLSTM`怎么办？
可以在定义TFLiteConverter时加上`map_bilstm_to_lstm=True`这个参数。

#### 如何转换带LSTM的模型？
由于我们转换的目标为TFLite，因此需要先了解一下在PyTorch和Tensorflow中LSTM分别是如何运行的。

使用TF2.X导出LSTM模型至Tensorflow Lite时，会将其翻译成`UnidirectionalLSTM`这个算子，其中的状态数据保存为一个`Variable`，即一个持久化的数据空间当中，每组mini-batch的状态会自动的做累积。这些状态量是不包含在模型的输入和输出之中的。

而在PyTorch中，LSTM含有一个可选的状态输入和状态输出，当不传入状态时，每次mini-batch的推理，初始隐层状态总是保持全0，这点与Tensorflow不同。

因此，为了能模拟Tensorflow这边的行为，在PyTorch侧导出LSTM模型时，请务必将LSTM的状态输入以及输出从模型输入、输出中删除。

那么，对于流式以及非流式的场景下，我们应该分别怎么去使用导出后的LSTM模型呢？

##### 非流式场景下
这种情况下，我们只需要将状态输入设置为0。所幸，Tensorflow Lite的Interpreter提供了一个方便的接口 [reset_all_variables](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter#reset_all_variables)。
所以，我们只需要在每次调用`invoke`之前，调用一次`reset_all_variables`即可。

##### 流式场景下
这种情况下，会稍许复杂一些，因为我们需要读写状态变量。我们可以使用Netron来打开生成的模型，定位到所有LSTM节点中，查看其中名称包含state的输入，例如对于单向LSTM状态量的属性名为`output_state_in`和`cell_state_in`，你可以展开后看到他们的kind为`Variable`。记住他们的位置（即`location`属性）。可以使用`tinynn.converter.utils.tflite.parse_lstm_states(tflite_path)`来获得模型中所有状态量的位置。

![image](https://user-images.githubusercontent.com/9998726/150492819-5f7d4f43-347e-4c1f-a700-72e77d95e9e9.png)

在使用Tensorflow Lite的Interpreter时，你只需要根据这些`location`，结合`get_tensor`和`set_tensor`方法就可以读或者写这些状态变量了。具体可参见[此处](https://github.com/alibaba/TinyNeuralNetwork/issues/29#issuecomment-1018292677)。

Note: 这些状态变量都是二维的，维度为`[batch_size, hidden_size或者input_size]`。所以在流式场景下，你只需要根据第一个维度对这些变量做拆分就可以了。

通常情况下，当隐层数量较大时（如128及以上）LSTM的模型在TFLite中会比较耗时。这种情况下，可以考虑使用动态范围量化来优化其性能，参见[dynamic.py](../examples/converter/dynamic.py)。

## 量化模型转换

#### 怎么把`ABS`、`SUM`、`SOFTMAX`、`LOG_SOFTMAX`和`BATCH_MATMUL`算子转换成定点？
首先，在定义`TFLiteConverter`时加上`rewrite_quantizable=True`这个参数。
其次，对于`SOFTMAX`、`LOG_SOFTMAX`，需要在定义`QATQuantizer`或者`PostQuantizer`时加上`set_quantizable_op_stats=True`这个参数。
