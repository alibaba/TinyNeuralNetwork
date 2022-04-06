import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

from examples.models.cifar10 import mobilenet
from tinynn.prune import OneShotChannelPruner
from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate
from tinynn.util.distributed_util import (
    deinit_distributed,
    get_device,
    init_distributed,
    is_distributed,
    is_main_process,
)
from tinynn.util.train_util import DLContext, train

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    # Whether the distributed backend is activated
    distributed = is_distributed(args)
    # Whether the code is running in the main process
    # Some tasks may be performed only when `main_process` is True to save time,
    # e.g. validating the model and saving weights
    main_process = is_main_process(args)

    init_distributed(args)
    device = get_device(args)
    print(f'Running on {device}')

    model = mobilenet.Mobilenet()
    model.load_state_dict(torch.load(mobilenet.DEFAULT_STATE_DICT))
    model.to(device=device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    pruner = OneShotChannelPruner(model, torch.ones(1, 3, 224, 224), args.config)

    # (Optional) A new config file with layer-level sparsity will be generated inplace
    # If you want to customize those generated content, you may do that before calling `.prune`
    pruner.generate_config(args.config)

    # Get the pruned untrained model
    pruner.prune()

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, distributed
    )
    validate(model, context)

    # (Optional) Use SyncBN to get better performance in distributed training
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # When adapting our framework to the existing training code, please make sure that the optimizer and the
    # lr_scheduler of the model is redefined using the weights of the new model.
    # e.g. If you use `get_optimizer` and `get_lr_scheduler` for constructing those objects, then you may write
    #   optimizer = get_optimizer(model)
    #   lr_scheduler = get_lr_scheduler(optimizer)

    context.max_epoch = 220
    context.criterion = nn.BCEWithLogitsLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    train(model, context, train_one_epoch, validate, distributed, main_process)

    # Cleanup DDP process group
    deinit_distributed(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    main_worker(args)
