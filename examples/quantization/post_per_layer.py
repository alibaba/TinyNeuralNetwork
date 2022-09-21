import argparse
import os
import sys
import copy

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, calibrate, validate
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.observer import HistogramObserverKL
from torch.nn.quantized import FloatFunctional

import torch.quantization as torch_q
from typing import List
from torch import Tensor

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def quantize_by_given_list(model: nn.Module, unique_name_list: List[str], alg: str = 'l2'):
    """Given a uname list of Op, using wrapper to quantify them.

    Args:
        model (nn.Module): A rewritten and fused model.
        alg: The ptq quantization algorithm.
        unique_name_list: The Op of model to be quantized.
    """
    backend = 'qnnpack'
    torch.backends.quantized.engine = backend
    qconfig = torch_q.get_default_qconfig(backend)
    if alg == 'kl':
        alg_asym_fq = HistogramObserverKL.with_args(reduce_range=False)
        qconfig = torch_q.QConfig(alg_asym_fq, qconfig.weight)
    ptq_model = copy.deepcopy(model)

    for name, mod in ptq_model.named_children():
        if name in unique_name_list:
            if isinstance(mod, torch.nn.quantized.FloatFunctional):
                new_mod = QuantFolatFunctionWrapper(mod)
            else:
                new_mod = torch.quantization.QuantWrapper(mod)
            setattr(new_mod, 'qconfig', qconfig)
            torch_q.propagate_qconfig_(new_mod, qconfig_dict=None)
            torch_q.add_observer_(new_mod)
            setattr(ptq_model, name, new_mod)
    return ptq_model


class QuantFolatFunctionWrapper(FloatFunctional):
    r"""A wrapper class that wraps the input FloatFunctional module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This implementation refers to torch_q.QuantWrapper, only valid for mul and add.
    """

    def __init__(self, ff_module):
        super(QuantFolatFunctionWrapper, self).__init__()
        self.add_module('quant_0', torch_q.QuantStub())
        self.add_module('quant_1', torch_q.QuantStub())
        self.add_module('dequant', torch_q.DeQuantStub())
        self.add_module('module', ff_module)
        self.train(ff_module.training)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.quant_0(x)
        y = self.quant_1(y)
        r = self.module.mul(x, y)
        r = self.dequant(r)
        return r

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.quant_0(x)
        y = self.quant_1(y)
        r = self.module.add(x, y)
        r = self.dequant(r)
        return r


def get_white_list_ptq():
    """Get the quantifiable Op type in PTQ."""
    if hasattr(torch_q, 'get_default_qconfig_propagation_list'):
        whitelist = torch_q.get_default_qconfig_propagation_list()
    elif hasattr(torch_q, 'get_qconfig_propagation_list'):
        whitelist = torch_q.get_qconfig_propagation_list()
    else:
        whitelist = torch_q.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
    # delete QuantStub and DeQuantStub
    if torch_q.QuantStub in whitelist:
        whitelist.remove(torch_q.QuantStub)
    if torch_q.DeQuantStub in whitelist:
        whitelist.remove(torch_q.DeQuantStub)
    return whitelist


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
        model_name = type(model).__name__

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # Get a rewritten and fused model
        quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'fuse_only': True})
        model = quantizer.quantize()

    # Initialize the model context, which include train/test dataset and calibrate/validate function.
    context = DLContext()
    device = get_device()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_iteration = 20
    model.to(device=device)

    # Use your validate function(return acc) to replace the validate function.
    acc_origin = validate(model, context)

    # Get the list of quantifiable ops' unique name, thr FloatFunctional ops is generated at rewriting step.
    unique_name_file_path = os.path.join(quantizer.work_dir, 'log', f'uname_{model_name}.yml')
    if not os.path.exists(unique_name_file_path):
        uname_config = dict()
        whitelist = get_white_list_ptq()
        for name, mod in model.named_children():
            if type(mod) in whitelist:
                uname_config[name] = type(mod).__name__
        with open(unique_name_file_path, 'w') as f:
            yaml.dump(uname_config, f, default_flow_style=False, Dumper=yaml.RoundTripDumper)
    with open(unique_name_file_path, 'r') as f:
        uname_config = yaml.load(f, Loader=yaml.RoundTripLoader)

    print("[TinyNeuralNetwork] Starting per_layer PTQ!")
    for layer_name, layer_cls in uname_config.items():
        # Add quantWrapper
        ptq_model = quantize_by_given_list(model, [layer_name])

        # Use DataParallel to speed up calibrating when possible.
        if torch.cuda.device_count() > 1:
            ptq_model = nn.DataParallel(ptq_model)

        # Move model to the appropriate device.
        ptq_model.to(device=device)
        context.device = device

        # Post quantization calibration, use your calibrate function to replace the calibrate function.
        ptq_model.eval()
        calibrate(ptq_model, context)

        with torch.no_grad():
            ptq_model.eval()
            ptq_model.cpu()

            # The step below converts the model to an actual quantized model, which uses the quantized kernels.
            ptq_model = torch.quantization.convert(ptq_model)

            if torch.cuda.device_count() > 1:
                ptq_model = ptq_model.module

            # Move model to cpu
            ptq_model.to(device=torch.device("cpu"))
            context.device = torch.device("cpu")

            # Use your validate function(return acc) to replace the validate function.
            acc = validate(ptq_model, context)
            print(f'[TinyNeuralNetwork] OP: {layer_name}({layer_cls}), Acc:{acc}({(acc - acc_origin):.4f})\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args()
    main_worker(args)
