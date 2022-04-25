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
from tinynn.graph.quantization import disable_fake_quant
from tinynn.graph.quantization.quantizer import BF16Quantizer
from tinynn.graph.tracer import model_tracer, trace
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate
from tinynn.util.train_util import DLContext, get_device, train


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # TinyNeuralNetwork provides a BF16Quantizer class that simulates computation with the bfloat16 datatype
        # The model returned by the `quantize` function is ready for bfloat16 training
        quantizer = BF16Quantizer(model, dummy_input, work_dir='out')
        bf16_model = quantizer.quantize()

    print(bf16_model)

    # Use DataParallel to speed up training when possible
    if torch.cuda.device_count() > 1:
        bf16_model = nn.DataParallel(bf16_model)

    # Move model to the appropriate device
    device = get_device()
    bf16_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_epoch = 1
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(bf16_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    train(bf16_model, context, train_one_epoch, validate, qat=True)

    with torch.no_grad():
        bf16_model.eval()
        bf16_model.cpu()

        # Disabling fake quants before exporting the model so that the model will behave the same as before.
        bf16_model.apply(disable_fake_quant)

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(bf16_model, dummy_input, tflite_path='out/qat_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)

    args = parser.parse_args()
    main_worker(args)
