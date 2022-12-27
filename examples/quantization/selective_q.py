import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import calibrate, get_dataloader, train_one_epoch, validate
from tinynn.util.train_util import DLContext, get_device, train

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # Generate quantized model file and weights
        # Note: this applies to `PostQuantizer` as well
        quantizer = QATQuantizer(model, dummy_input, work_dir='out')
        q_model = quantizer.quantize()

        # By default, we tend to quantize as much layers as possible. But you are free to make some modifications.
        print(
            f'The layerwise configuration file is generated under {os.path.abspath(quantizer.work_dir)}\nIt shows'
            ' whether each layer is quantized or not. The hidden layers are layers that cannot be quantized or fused'
            ' into other layers.\nYou can modify it to generate the corresponding model for quantization rewrite.'
        )
        input(
            "If no changes are made to the file, please press Enter to continue. Otherwise, press Ctrl+C, make the"
            " changes and rerun this script"
        )

    # Use DataParallel to speed up training when possible
    if torch.cuda.device_count() > 1:
        q_model = nn.DataParallel(q_model)

    # Move model to the appropriate device
    device = get_device()
    q_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_epoch = 3
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(q_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    if isinstance(quantizer, QATQuantizer):
        # Quantization-aware training
        train(q_model, context, train_one_epoch, validate, qat=True)
    else:
        # Post quantization calibration
        calibrate(q_model, context)

    with torch.no_grad():
        q_model.eval()
        q_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        q_model = quantizer.convert(q_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with symmetric quantization (int8),
        # you may specify `asymmetric=False` in the following line.
        converter = TFLiteConverter(q_model, dummy_input, tflite_path='out/q_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)

    args = parser.parse_args()
    main_worker(args)
