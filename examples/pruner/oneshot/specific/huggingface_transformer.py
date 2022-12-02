"""
The pre-training code of vit comes from https://github.com/nateraw/huggingface-vit-finetune .
"""

import sys
import argparse

sys.path.append('../../../../')

import torch
import torch.nn as nn

from tinynn.util.util import import_from_path
from tinynn.graph.tracer import import_patcher
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.train_util import DLContext, get_device

with import_patcher():
    from transformers import BertModel


class ViTWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x).logits


def main_worker(args):
    print("###### TinyNeuralNetwork quick start for beginner ######")

    model = BertModel.from_pretrained('bert-base-uncased')
    model.cpu()
    device = get_device()

    dummy_input_0 = torch.ones((1, 8), dtype=torch.int64)
    dummy_input_1 = torch.ones((1, 8), dtype=torch.int64)
    dummy_input_2 = torch.ones((1, 8), dtype=torch.int64)

    dummy_input = (dummy_input_0, dummy_input_1, dummy_input_2)

    context = DLContext()
    context.device = device

    pruner = OneShotChannelPruner(
        model, dummy_input, {"sparsity": 3 / 12, "metrics": "l2_norm", "exclude_ops": [nn.LayerNorm]}
    )
    pruner.prune()  # Get the pruned model
    pruner.graph.generate_code('out/vit.py', 'out/vit.pth', 'vit')

    model = import_from_path('out.vit', "out/vit.py", "vit")()
    model.load_state_dict(torch.load("out/vit.pth"))
    model.eval()
    model(*dummy_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to cifar10 dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--amp', type=bool, default=True)

    args = parser.parse_args()
    main_worker(args)
