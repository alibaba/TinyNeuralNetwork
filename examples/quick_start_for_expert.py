import os
import copy
import argparse

import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.prune import OneShotChannelPruner
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, train_one_epoch_distill, validate
from tinynn.util.train_util import DLContext, get_device, train
from tinynn.util.util import set_global_log_level

from examples.models.cifar10 import mobilenet

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    print("###### TinyNeuralNetwork quick start for expert ######")

    # If you encounter any problems, please set the global log level to `DEBUG`, which may make it easier to debug.
    # set_global_log_level("DEBUG")

    model = mobilenet.Mobilenet()
    model.load_state_dict(torch.load(mobilenet.DEFAULT_STATE_DICT))

    device = get_device()
    model.to(device=device)

    if args.distillation:
        teacher = copy.deepcopy(model)

    if args.parallel:
        model = nn.DataParallel(model)

    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)

    print("Validation accuracy of the original model")
    validate(model, context)

    print("Start pruning the model")
    # If you need to set the sparsity of a single operator, then you may refer to the examples in `examples/pruner`.
    pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.75, "metrics": "l2_norm"})

    st_flops = pruner.calc_flops()
    pruner.prune()  # Get the pruned model

    print("Validation accuracy of the pruned model")
    validate(model, context)

    ed_flops = pruner.calc_flops()
    print(f"Pruning over, reduced FLOPS {100 * (st_flops - ed_flops) / st_flops:.2f}%  ({st_flops} -> {ed_flops})")

    print("Start finetune the pruned model")
    # In our experiments, using the same learning rate configuration as the one used during training from scratch
    # leads to a higher final model accuracy.
    context.max_epoch = 220
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    context.scheduler = CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    if args.warmup:
        print("Use warmup")
        context.warmup_iteration = len(context.train_loader) * 10  # warmup 10 epoch
        context.warmup_scheduler = CyclicLR(
            context.optimizer, base_lr=0, max_lr=0.1, step_size_up=context.warmup_iteration
        )

    if args.distillation:
        # The utilization of distillation may leads to better accuracy at the price of longer training time.
        print("Use distillation")
        context.custom_args = {'distill_A': 0.3, 'distill_T': 6, 'distill_teacher': teacher}
        train(model, context, train_one_epoch_distill, validate)
    else:
        train(model, context, train_one_epoch, validate)

    print("Start preparing the model for quantization")
    # We provides a QATQuantizer class that may rewrite the graph for and perform model fusion for quantization
    # The model returned by the `quantize` function is ready for QAT training
    quantizer = QATQuantizer(model, dummy_input, work_dir='out')
    qat_model = quantizer.quantize()

    print("Start quantization-aware training")
    qat_model.to(device=device)

    # Use DataParallel to speed up training when possible
    if args.parallel:
        qat_model = nn.DataParallel(qat_model)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    context.max_epoch = 20
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(qat_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    # Quantization-aware training
    train(qat_model, context, train_one_epoch, validate, qat=True)

    print("Start converting the model to TFLite")
    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = quantizer.convert(qat_model)

        # When converting quantized models to TFLite, please ensure the quantization backend is QNNPACK.
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to cifar10 dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--distillation', type=bool, default=True)

    args = parser.parse_args()
    main_worker(args)
