# Vision Transformer quantization
Original full precision checkpoints without downstream training were downloaded through [hugging face](https://huggingface.co/google/vit-base-patch16-224).

- Additional Setup Dependencies
```shell
pip install transformers==4.26.0
```

# DataSet
This evaluation was designed for the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012), The dataset directory is expected to have 3 subdirectories: train, valid, and test. We use samples of train dataset to do ptq calibrating and valid dataset to evaluate model performance.

Each of the {train, valid, test} directories is then expected to have 1000 subdirectories, each containing the images from the 1000 classes present in the ILSVRC2012 dataset, such as in the example below:
```
train/
  ├── n01440764
  │   ├── n01440764_10026.JPEG
  │   ├── n01440764_10027.JPEG
  │   ├── ......
  ├── ......
  val/
  ├── n01440764
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   ├── ......
  ├── ......
```

# Usage
see [ptq example](vit_post.py).


# Quantization Accuracy Results

We use post quantization to get a real INT8 VIT, the accuracy@1 result is as below:

| Model         | Top1 Acc (%) | Mixed Quantization Configuration                                         |
|---------------|--------------|--------------------------------------------------------------------------|
| Origin(FP32)  | 81.43        | -                                                                        |
| INT8          | 0            | all                                                                      |
| INT8(Mixed_1) | 78.49(-2.94) | all, except residual additions                                           |
| INT8(Mixed_2) | 80.94(-0.49) | all, except residual additions , <br/>two most quantization-sensitive fc |

1. We used [per-layer quantization analysis tool](../../layerwise_ptq_analysis.py) and observed that residual additions have a notable impact on quantization accuracy,
leading to a significant drop in accuracy. To address this, we maintained floating-point calculation for all residual additions.
While this helped to restore some accuracy, we found that there was still a considerable drop in accuracy.(-2.94).

2. When using [quantization error analysis tools](../../post_error_anaylsis.py), We noticed that the output activation of the MLP contains numerous outliers, which can considerably reduce the accuracy of quantization.
Consequently, we identified the two most sensitive fully connected layers and maintained floating-point calculations for them. As a result, the accuracy was further improved(-0.49).

3. Apart from the mixed-precision quantized operators for floating-point computations mentioned above,
the `LayerNorm` and `Gelu` operators also retain floating-point computations due to lack of support from the backend.
