import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

from examples.models.cifar10 import mobilenet
from tinynn.prune import NetAdaptPruner
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate
from tinynn.util.train_util import DLContext, get_device, train


def load_model():
    model = mobilenet.Mobilenet()
    model.load_state_dict(torch.load(mobilenet.DEFAULT_STATE_DICT))

    return model


def load_pruner(model, args):
    config = NetAdaptPruner.load_config(args.config)
    if torch.cuda.is_available():
        workers = args.workers // torch.cuda.device_count() * 2
    else:
        workers = 1

    context = DLContext()
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, workers)
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), config['netadapt_lr'], momentum=0.9, weight_decay=1e-4)

    # The train_one_epoch function provided by the user must support the
    # max_iteration parameter (stop after training for n iterations)
    context.train_func = train_one_epoch
    context.validate_func = validate
    pruner = NetAdaptPruner(model, torch.ones(1, 3, 224, 224), config, context)

    return pruner


def main_worker(args):
    model = load_model()
    pruner = load_pruner(model, args)

    # Don't use any parallel modules here, since NetAdapt is already running on multiple GPUs.

    # (Optional) Generate config with optional arguments
    # To make the changes effective, you may need to rerun the script
    pruner.generate_config(args.config)

    # (Optional) Restore task from specified iteration
    # pruner.restore(iteration=5)

    # Model compress
    pruner.prune()

    # (Optional) After calling prune, you may get multiple models with various FLOPs and accuracy.
    # Pick one as you like, then you could restore the state of that model.
    # model = load_model()
    # pruner = load_pruner(model, args)
    # pruner.restore(iteration=50)

    print("Start finetune")

    if args.distributed:
        model = torch.nn.parallel.DataParallel(model)

    device = get_device()
    model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
    validate(model, context)

    # When adapting our framework to the existing training code, please make sure that the optimizer and the
    # lr_scheduler of the model is redefined using the weights of the new model.
    # e.g. If you use `get_optimizer` and `get_lr_scheduler` for constructing those objects, then you may write
    #   optimizer = get_optimizer(model)
    #   lr_scheduler = get_lr_scheduler(optimizer)

    # Fine tune
    context.max_epoch = 220
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    train(model, context, train_one_epoch, validate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--distributed', type=bool, default=True)

    args = parser.parse_args()
    main_worker(args)
