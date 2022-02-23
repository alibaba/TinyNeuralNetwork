import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('../../../')

from examples.models.cifar10 import mobilenet
from tinynn.prune import ADMMPruner
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    config = ADMMPruner.load_config(args.config)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, distributed
    )
    context.criterion = nn.CrossEntropyLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), config['admm_lr'], momentum=0.9, weight_decay=2.5e-4)
    context.train_func = train_one_epoch
    context.validate_func = validate
    pruner = ADMMPruner(model, torch.ones(1, 3, 224, 224), config, context)

    # (Optional) Generate config with layer-wise sparsity
    # To make the changes effective, you may need to rerun the script
    pruner.generate_config(args.config)

    pruner.prune()

    # (Optional) Use SyncBN to get better performance in distributed training
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, distributed
    )
    validate(model, context)

    # fine tune
    context.max_epoch = 220
    context.criterion = nn.CrossEntropyLoss()
    context.optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=2.5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    train(model, context, train_one_epoch, validate, distributed, main_process)

    # Cleanup DDP process group
    deinit_distributed(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    main_worker(args)
