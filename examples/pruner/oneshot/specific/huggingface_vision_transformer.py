"""
The pre-training code of vit comes from https://github.com/nateraw/huggingface-vit-finetune .
"""

import sys
import argparse

sys.path.append('../../../../')

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR

from tinynn.converter import TFLiteConverter
from tinynn.util.util import import_from_path
from tinynn.util.cifar10 import train_one_epoch
from tinynn.graph.tracer import patch_helper, no_catch, model_tracer
from tinynn.util.cifar10 import get_dataloader, validate
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.train_util import train, DLContext, get_device

with model_tracer():
    with patch_helper(wrap_creation_funcs=False, wrap_funcs=True, wrap_modules=False):
        with no_catch():
            from transformers import ViTForImageClassification


class ViTWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x).logits


def main_worker(args):
    print("###### TinyNeuralNetwork quick start for beginner ######")

    model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    model = ViTWrapper(model)
    model.cpu()
    device = get_device()

    # Provide a viable input for the model
    dummy_input = torch.rand((1, 3, 224, 224))

    context = DLContext()
    context.device = device

    # The pretrained model uses 0.5 for std and mean
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )

    print("Validation accuracy of the original model")
    # validate(model, context)

    print("\n###### Start pruning the model ######")
    # Only prune the inside fc of the Transformer, without changing the feature dimension between the Transformers.
    # Each transformer has 12 heads, so the pruning rate must be N/12 (N<12)
    pruner = OneShotChannelPruner(
        model, dummy_input, {"sparsity": 2 / 12, "metrics": "l2_norm", "exclude_ops": [nn.LayerNorm]}
    )

    pruner.prune()  # Get the pruned model
    pruner.graph.generate_code('out/vit.py', 'out/vit.pth', 'vit')

    model = import_from_path('out.vit', "out/vit.py", "vit")()
    model.load_state_dict(torch.load("out/vit.pth"))
    model.eval()

    model.to(device=device)

    if args.parallel:
        print("use data parallel")
        model = nn.DataParallel(model)

    print("Validation accuracy of the pruned model")
    validate(model, context)

    warmup_epoch = 1
    context.max_epoch = 5
    context.print_freq = 5
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size // 2, args.workers, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )

    train_iter = len(context.train_loader)
    context.criterion = nn.CrossEntropyLoss()
    context.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    context.iter_scheduler = CosineAnnealingLR(
        context.optimizer, T_max=train_iter * (context.max_epoch - warmup_epoch) + 1, eta_min=0
    )

    print("Use lr warmup")
    context.warmup_iteration = train_iter * warmup_epoch
    context.warmup_scheduler = CyclicLR(
        context.optimizer, base_lr=0, max_lr=2e-5, cycle_momentum=False, step_size_up=context.warmup_iteration
    )
    context.warmup_scheduler.step()

    if args.amp:
        print("Use automatic mixed precision")
        context.grad_scaler = GradScaler()

    print("\n###### Start finetune the pruned model ######")
    """
    Original accuracy: 98.52
    1. Pruning 1/12: 95.21 -> 98.68 (warmup_epoch=1， max_epoch=5)
    2. pruning 2/12: 75.20 -> 98.19 (warmup_epoch=1， max_epoch=5)
    This is just a simple example, you can further improve the accuracy by
    optimizing hyperparameters, distillation, optimizing the loss function, etc.
    """
    train(model, context, train_one_epoch, validate)

    print("\n###### Start converting the model to TFLite ######")
    with torch.no_grad():
        model.eval()
        model.cpu()

        converter = TFLiteConverter(model, dummy_input, tflite_path='out/vit.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to cifar10 dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--amp', type=bool, default=True)

    args = parser.parse_args()
    main_worker(args)
