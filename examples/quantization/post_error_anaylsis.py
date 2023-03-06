import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, calibrate
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis, get_weight_dis


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        device = get_device()

        # When the weight distributions fluctuates greatly, CLE may significantly increase the quantization accuracy.
        if args.cle:
            model = cross_layer_equalize(model, dummy_input, device, cle_iters=2)

        # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
        # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
        quantizer = QATQuantizer(
            model, dummy_input, work_dir='out', config={'override_qconfig_func': set_ptq_fake_quantize}
        )
        ptq_model = quantizer.quantize()

    print(ptq_model)

    # Use DataParallel to speed up calibrating when possible
    if torch.cuda.device_count() > 1:
        ptq_model = nn.DataParallel(ptq_model)

    # Move model to the appropriate device
    ptq_model.to(device=device)
    ptq_model.eval()

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_iteration = 100

    # Post quantization calibration
    ptq_model.apply(torch.quantization.disable_fake_quant)
    ptq_model.apply(torch.quantization.enable_observer)
    calibrate(ptq_model, context)

    # Disable observer and enable fake quantization to validate model with quantization error
    ptq_model.apply(torch.quantization.disable_observer)
    ptq_model.apply(torch.quantization.enable_fake_quant)
    dummy_input.to(device)
    ptq_model(dummy_input)
    print(ptq_model)

    # Perform quantization error analysis with real dummy input
    dummy_input_real = next(iter(context.train_loader))[0][:1]

    # --------weight visualization usage-------------
    # Draw the weight distribution for all layers of model.
    get_weight_dis(ptq_model, save_path='out')

    # --------quantization error analysis usage-------------
    # The error is accumulated by directly giving the difference in layer output
    # between the quantized model and the floating model. If you want a quantized model with high accuracy,
    # the layers closest to the final output should be less than 10%, which means the final
    # layer's cosine similarity should be greater than 90%.
    graph_error_analysis(ptq_model, dummy_input_real, metric='cosine')

    # We quantize each layer separately and compare the difference
    # between the original output and the output with quantization error for each layer,
    # which is used to calculate the quantization sensitivity of each layer.
    layer_error_analysis(ptq_model, dummy_input_real, metric='cosine')
    # -----------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cle', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args()
    main_worker(args)
