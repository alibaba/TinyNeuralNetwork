import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate
from tinynn.util.train_util import DLContext, get_device, train

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # generate qat model file and weights
        quantizer = QATQuantizer(model, dummy_input, work_dir='out')
        quantizer.quantize()

    # Modify the generated qat model file and mix it according to your needs
    # The modification method can refer to mobilenet_qat.py
    with model_tracer():
        from mobilenet_mixed_qat import Mobilenet_qat

        qat_mobilenet = Mobilenet_qat()
        qat_mobilenet.load_state_dict(torch.load("./out/mobilenet_qat.pth"))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # generate qat model file and weights
        quantizer = QATQuantizer(qat_mobilenet, dummy_input, config={'rewrite_graph': False}, work_dir='out')
        qat_model = quantizer.quantize()

    # Use DataParallel to speed up training when possible
    if torch.cuda.device_count() > 1:
        qat_model = nn.DataParallel(qat_model)

    # Move model to the appropriate device
    device = get_device()
    qat_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_epoch = 3
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
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with symmetric quantization (int8),
        # you may specify `asymmetric=False` in the following line.
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)

    args = parser.parse_args()
    main_worker(args)
