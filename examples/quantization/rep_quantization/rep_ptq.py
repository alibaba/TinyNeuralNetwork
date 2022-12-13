import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

import torch
import torch.nn as nn

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter

from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.quantization.cross_layer_equalization import cross_layer_equalize
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, calibrate
from tinynn.util.train_util import DLContext, get_device


def main_worker(args):
    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    # Provide your pretrain model
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

    # Define the train_related context
    device = get_device()
    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)

    # if your pretrained model is Rep_style_deploy, use the following line to do bn restore to do High Bias Absorb.
    # model = model_restore_bn(model, device, calibrate, context)
    # Do CLE, if weight has some outliers which is hard to quantize, considering trying CLE.
    model = cross_layer_equalize(model, dummy_input, device)

    # Now you get a model whose weights and activations is easy to quantize, continue to do ptq.
    with model_tracer():
        # More information for PostQuantizer initialization, see `examples/quantization/post.py`.
        quantizer = PostQuantizer(model, dummy_input, work_dir='out')
        ptq_model = quantizer.quantize()

    print(ptq_model)

    # Use DataParallel to speed up calibrating when possible
    if torch.cuda.device_count() > 1:
        ptq_model = nn.DataParallel(ptq_model)

    # Move model to the appropriate device
    ptq_model.to(device=device)
    # Set calibrate iteration nums
    context.max_iteration = 100

    # Post quantization calibration
    calibrate(ptq_model, context)

    with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model = torch.quantization.convert(ptq_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(ptq_model, dummy_input, tflite_path='out/rep_ptq.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)

    args = parser.parse_args()
    main_worker(args)
