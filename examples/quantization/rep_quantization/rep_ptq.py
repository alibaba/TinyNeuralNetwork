import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as torch_q

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer, PostQuantizer, load_processed_ptq_rules
from tinynn.graph.tracer import model_tracer, trace
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate, calibrate
from tinynn.util.train_util import DLContext, get_device, train

from tinynn.graph.quantization.cross_layer_equalization import cross_layer_equalize
from tinynn.util.bn_restore import model_restore_bn


def main_worker(args):
    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    # We use BN_folded MobileNetv1 to simulate reparameterized MobileOne.
    # You can also directly use the reparameterized model of MobileOne_deploy(or other rep_deploy_model).
    model = reparameterize_model_for_deploy(dummy_input)

    # Do CLE(Optional), if weight of conv_fused_bn has some outliers which is hard to quantize, you can try the CLE.
    with torch.no_grad():
        cross_layer_equalize(model, dummy_input)

    # Continue to do ptq.
    with model_tracer():
        # TinyNeuralNetwork provides a PostQuantizer class that may rewrite the graph for and perform model fusion for
        # quantization. The model returned by the `quantize` function is ready for quantization calibration.
        # By default, the rewritten model (in the format of a single file) will be generated in the working directory.
        # You may also pass some custom configuration items through the argument `config` in the following line. For
        # example, if you have a quantization-rewrite-ready model (e.g models in torchvision.models.quantization),
        # then you may use the following line.
        #   quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'rewrite_graph': False})
        # Alternatively, if you have modified the generated model description file and want the quantizer to load it
        # instead, then use the code below.
        #     quantizer = PostQuantizer(
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
        # In addition, we support additional ptq algorithms including kl-divergence, the usage is shown as below:
        #       quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'algorithm': 'kl'})

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


def reparameterize_model_for_deploy(dummy_input):
    """The helper function to get conv_bn folded model."""
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
