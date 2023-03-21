# This script use PTQFakeQuantize to simulate quantization error,
# so that we can validate model with quantization error using PyTorch backend on gpu.
# Currently, Pytorch quantization backend only support for interface on cpu.
import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../../'))

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

yaml_ = yaml.YAML()

import torch
import torch.nn as nn
import torch.quantization as torch_q

from examples.quantization.specific.util.imagenet_util import get_dataloader, validate, calibrate
from mobileone_origin import get_model
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis, get_weight_dis


def ptq_mobileone(args):
    device = get_device()
    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dummy_input = torch.rand((1, 3, 224, 224))
    # Download the MobileOne model pth from https://github.com/apple/ml-mobileone
    model = get_model(variant='s1', model_pth=args.model_pth_dir)

    context.max_iteration = 20
    model.eval()
    # When the weight distributions fluctuates greatly, CLE may significantly increase the quantization accuracy.
    if args.cle:
        model = cross_layer_equalize(model, dummy_input, device, cle_iters=4)
        validate(model, context)

    # Draw the weight distribution for all layers of model.
    get_weight_dis(model, save_path='out')

    with model_tracer():
        # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
        # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
        quantizer = QATQuantizer(
            model, dummy_input, work_dir='out', config={'override_qconfig_func': set_ptq_fake_quantize}
        )
        ptq_model = quantizer.quantize()

    if torch.cuda.device_count() > 1:
        ptq_model = nn.DataParallel(ptq_model)

    ptq_model.to(device=device)
    # Set number of iteration for calibration
    context.max_iteration = 20

    # Post quantization calibration
    ptq_model.apply(torch_q.disable_fake_quant)
    ptq_model.apply(torch_q.enable_observer)
    calibrate(ptq_model, context)

    # Disable observer and enable fake quantization to validate model with quantization error
    ptq_model.apply(torch_q.disable_observer)
    ptq_model.apply(torch_q.enable_fake_quant)
    dummy_input.to(device)
    ptq_model(dummy_input)
    print(ptq_model)

    # Perform quantization error analysis with real dummy input
    dummy_input_real = next(iter(context.train_loader))[0][:1]

    # The error is accumulated by directly giving the difference in layer output
    # between the quantized model and the floating model. If you want a quantized model with high accuracy,
    # the layers closest to the final output should be less than 10%, which means the final
    # layer's cosine similarity should be greater than 90%.
    graph_error_analysis(ptq_model, dummy_input_real, metric='cosine')

    # We quantize each layer separately and compare the difference
    # between the original output and the output with quantization error for each layer,
    # which is used to calculate the quantization sensitivity of each layer.
    layer_error_analysis(ptq_model, dummy_input_real, metric='cosine')

    # validate the model with quantization error via fake quantization
    validate(ptq_model, context)

    with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        if isinstance(ptq_model, nn.DataParallel):
            ptq_model = ptq_model.module

        ptq_model = torch.quantization.convert(ptq_model)
        context.device = torch.device('cpu')

        # validate the quantized model
        validate(ptq_model, context)

        torch.backends.quantized.engine = quantizer.backend
        converter = TFLiteConverter(model, dummy_input, tflite_path='out/mobileone_ptq.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/imagenet", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cle', type=bool, default=True)
    parser.add_argument('--model-pth-dir', type=str, default='out')

    args = parser.parse_args()
    ptq_mobileone(args)
