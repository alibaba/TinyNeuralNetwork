import argparse
import os
import sys

import torch

from tinynn.prune.identity_pruner import IdentityChannelPruner

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../'))

from examples.models.cifar10 import mobilenet
from tinynn.util.cifar10 import get_dataloader, validate
from tinynn.util.distributed_util import (
    get_device,
    init_distributed,
    is_distributed,
)
from tinynn.util.train_util import DLContext

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def main_worker(args):
    # Whether the distributed backend is activated
    distributed = is_distributed(args)
    # Whether the code is running in the main process
    # Some tasks may be performed only when `main_process` is True to save time,
    # e.g. validating the model and saving weights

    init_distributed(args)
    device = get_device(args)
    print(f'Running on {device}')

    model = mobilenet.Mobilenet()
    model.load_state_dict(torch.load(mobilenet.DEFAULT_STATE_DICT))
    model.to(device=device)

    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, distributed
    )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    print("Validation accuracy of the original model")
    validate(model, context)

    # You can specify that the pruned channel is a multiple of n by configuring config={"multiple":n}
    pruner = IdentityChannelPruner(model, torch.ones(1, 3, 224, 224), config={"multiple": 8})

    st_flops = pruner.calc_flops()
    pruner.prune()  # Get the pruned model

    print("Validation accuracy of the pruned model")
    validate(model, context)
    ed_flops = pruner.calc_flops()
    print(f"Pruning over, reduced FLOPS {100 * (st_flops - ed_flops) / st_flops:.2f}%  ({st_flops} -> {ed_flops})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    main_worker(args)
