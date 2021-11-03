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
