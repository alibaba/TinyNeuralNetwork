import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

import functools

import torch
import torch.nn as nn
import torch.quantization as torch_q

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer, PostQuantizer, load_processed_ptq_rules
from tinynn.graph.tracer import model_tracer, trace
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate, calibrate
from tinynn.util.train_util import DLContext, get_device, train

from tinynn.graph.quantization.cross_layer_equalization import cross_layer_equalize


def main_worker(args):
    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    # We use BN_fused MobileNetV1 to simulate MobileOne_deploy which be re-parameterized.
    # You can directly use the re-parameterized_to_deploy model.
    model = mobilenet_fused_bn(dummy_input)

    # Do CLE(Optional), if weight of conv_bn_fused has some outliers which is hard to quantize, considering trying CLE.
    cross_layer_equalize(model, dummy_input)

    # Continue to do ptq.
    with model_tracer():
        # More information for PostQuantizer initialization, see `examples/quantization/post.py`.
        quantizer = PostQuantizer(model, dummy_input, work_dir='out')
        ptq_model = quantizer.quantize()

    print(ptq_model)

    # Use DataParallel to speed up calibrating when possible
    if torch.cuda.device_count() > 1:
        ptq_model = nn.DataParallel(ptq_model)

    # Move model to the appropriate device
    device = get_device()
    ptq_model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
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


def mobilenet_fused_bn(dummy_input):
    """
    The helper function to get conv_bn fused mobilenet. Since the BN of Rep-model(e.g. RepVGG/MobileOne) is fused
    into the conv when the parameters are merged. We fuse the BN of MobileNet into the conv to simulate MobileOne.
    """
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    with model_tracer():
        model.eval()
        quantizer = PostQuantizer(
            model,
            dummy_input,
            work_dir='../out',
            config={'rewrite_graph': False, 'force_overwrite': False, 'fuse_only': True},
        )
        graph = trace(quantizer.model, quantizer.dummy_input)
        graph.quantized = True
        for node in graph.forward_nodes:
            node.quantized = True
        custom_data = ([], set())
        processed_rules = load_processed_ptq_rules()
        processed_rules = {nn.BatchNorm2d: processed_rules[nn.BatchNorm2d]}
        is_fusable = functools.partial(quantizer.is_fusable, current_rules=processed_rules, graph=graph)
        graph.filter_forward_nodes(is_fusable, custom_data, reverse=True)
        quant_list = custom_data[0]
        for quant_nodes in quant_list:
            torch_q.fuse_modules(graph.module, quant_nodes, inplace=True)
        fused_model = graph.module
    return fused_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=192)

    args = parser.parse_args()
    main_worker(args)
