import argparse
import os

import torch
import torch.nn as nn
from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, calibrate
from tinynn.util.train_util import DLContext, get_device


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # TinyNeuralNetwork provides a PostQuantizer class that may rewrite the graph for and perform model fusion for quantization
        # The model returned by the `quantize` function is ready for quantization calibration
        # By default, the rewritten model (in the format of a single file) will be generated in the working directory.
        # You may also pass some custom configuration items through the argument `config` in the following line.
        # For example, if you have a quantization-calibration-ready model (e.g models in torchvision.models.quantization),
        # then you may use the following line.
        #   quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'rewrite_graph': False})
        # Alternatively, if you have modified the generated model description file and want the quantizer to load it instead,
        # then use the code below.
        #   quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'force_overwrite': False, 'is_input_quantized': None})
        # The `is_input_quantized` in the previous line is a flag on the input tensors whether they are quantized or not,
        # which can be None (False for all inputs) or a list of booleans that corresponds to the inputs.

        quantizer = PostQuantizer(model, dummy_input, work_dir='out')
        qat_model = quantizer.quantize()

    print(qat_model)

    # Use DataParallel to speed up calibrating when possible
    if torch.cuda.device_count() > 1:
        qat_model = nn.DataParallel(qat_model)

    # Move model to the appropriate device
    device = get_device()
    qat_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_iteration = 100

    # Post quantization calibration
    calibrate(qat_model, context)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = torch.quantization.convert(qat_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args()
    main_worker(args)
