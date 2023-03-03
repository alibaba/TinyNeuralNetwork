import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn
import torch.optim as optim

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate, calibrate
from tinynn.util.train_util import DLContext, get_device, train
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize
from tinynn.util.bn_restore import model_restore_bn


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        device = get_device()
        context = DLContext()
        context.device = device
        context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)

        # When the weight distributions fluctuates greatly, CLE may significantly increase the quantization accuracy.
        if args.cle:
            model = cross_layer_equalize(model, dummy_input, device)

        # You may want to insert `BatchNorm` layers after `Conv` layers for the following case.
        # 1. For RepVGG-like model. With reparameterization, the model is hard to train without `BatchNorm` layers.
        # 2. For QAT with models that applied CLE. Since CLE tries to fuse `Conv` and `BatchNorm` layers, it would
        #    better to restore it to the original state.
        if args.bn_restore:
            model = model_restore_bn(model, get_device(), calibrate, context)

        # TinyNeuralNetwork provides a QATQuantizer class that may rewrite the graph for and perform model fusion for
        # quantization. The model returned by the `quantize` function is ready for QAT.
        # By default, the rewritten model (in the format of a single file) will be generated in the working directory.
        # You may also pass some custom configuration items through the argument `config` in the following line. For
        # example, if you have a QAT-ready model (e.g models in torchvision.models.quantization),
        # then you may use the following line.
        #   quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'rewrite_graph': False})
        # Alternatively, if you have modified the generated model description file and want the quantizer to load it
        # instead, then use the code below.
        #     quantizer = QATQuantizer(
        #         model, dummy_input, work_dir='out', config={'force_overwrite': False, 'is_input_quantized': None}
        #     )
        # The `is_input_quantized` in the previous line is a flag on the input tensors whether they are quantized or
        # not, which can be None (False for all inputs) or a list of booleans that corresponds to the inputs.
        # Also, we support multiple qschemes for quantization preparation. There are several common choices.
        #   a. Asymmetric uint8. (default) config={'asymmetric': True, 'per_tensor': True}
        #      The is the most common choice and also conforms to the legacy TFLite quantization spec.
        #   b. Asymmetric int8. config={'asymmetric': True, 'per_tensor': False}
        #      The conforms to the new TFLite quantization spec. In legacy TF versions, this is usually used in post
        #      quantization. Compared with (a), it has support for per-channel quantization in supported kernels
        #      (e.g Conv), while (a) does not.
        #   c. Symmetric int8. config={'asymmetric': False, 'per_tensor': False}
        #      The is same to (b) with no offsets, which may be used on some low-end embedded chips.
        #   d. Symmetric uint8. config={'asymmetric': False, 'per_tensor': True}
        #      The is same to (a) with no offsets. But it is rarely used, which just serves as a placeholder here.

        quantizer = QATQuantizer(model, dummy_input, work_dir='out')
        qat_model = quantizer.quantize()

    print(qat_model)

    # Use DataParallel to speed up training when possible
    if torch.cuda.device_count() > 1:
        qat_model = nn.DataParallel(qat_model)

    # Move model to the appropriate device
    qat_model.to(device=device)

    # When adapting our framework to the existing training code, please make sure that the optimizer and the
    # lr_scheduler of the model is redefined using the weights of the new model.
    # e.g. If you use `get_optimizer` and `get_lr_scheduler` for constructing those objects, then you may write
    #   optimizer = get_optimizer(qat_model)
    #   lr_scheduler = get_lr_scheduler(optimizer)
    context.max_epoch = 30
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(qat_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    # Quantization-aware training
    train(qat_model, context, train_one_epoch, validate, qat=True)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = quantizer.convert(qat_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--cle', type=bool, default=False)
    parser.add_argument('--bn-restore', type=bool, default=False)

    args = parser.parse_args()
    main_worker(args)
