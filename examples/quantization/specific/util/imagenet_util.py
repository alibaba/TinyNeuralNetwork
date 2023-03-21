import os
import time
import typing

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tinynn.util.train_util import AverageMeter, DLContext


def get_dataloader(
    data_path: str,
    img_size: int = 224,
    batch_size: int = 256,
    worker: int = 16,
    distributed: bool = False,
    download: bool = False,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2023, 0.1994, 0.2010),
) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Constructs the dataloaders for training and validating

    Args:
        data_path (str): The path of the dataset
        img_size (int, optional): The size of the image. Defaults to 224.
        batch_size (int, optional): The batch size of the dataloader. Defaults to 128.
        worker (int, optional): The number of workers. Defaults to 4.
        distributed (bool, optional): Whether to use DDP. Defaults to False.
        download (bool, optional): Whether to download the dataset. Defaults to False.
        mean (tuple, optional): Normalize mean
        std (tuple, optional): Normalize std

    Returns: typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The dataloaders for training and
    validating
    """
    assert download is False, "download=True is not implemented for ImageNet"
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "validation")
    assert os.path.exists(train_path)
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=(train_sampler is None),
        num_workers=worker,
        pin_memory=True,
        sampler=train_sampler,
    )

    assert os.path.exists(val_path)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            val_path,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker,
        pin_memory=True,
    )

    return train_loader, val_loader


def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = torch.sum(target == output).item()
    acc = acc / output.size(0) * 100
    return acc


def train_one_epoch(model, context: DLContext):
    """Train the model for one epoch

    Args:
        model: The model to be trained
        context (DLContext): The context object
    """

    avg_batch_time = AverageMeter()
    avg_data_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_acc = AverageMeter()

    model.to(device=context.device)
    model.train()

    end = time.time()
    for i, (image, label) in enumerate(context.train_loader):
        if context.max_iteration is not None and i >= context.max_iteration:
            break

        avg_data_time.update(time.time() - end)
        image = image.to(device=context.device)
        # compute output and loss
        output = model(image)

        # For loss computation of CrossEntropyLoss
        label = label.to(device=context.device)
        loss = context.criterion(output, label)

        context.optimizer.zero_grad()
        loss.backward()
        context.optimizer.step()

        avg_losses.update(loss.item(), image.size(0))
        avg_acc.update(compute_accuracy(output, label), image.size(0))
        avg_batch_time.update(time.time() - end)
        end = time.time()

        if i % context.print_freq == 0:
            current_lr = 0.0
            for param_group in context.optimizer.param_groups:
                current_lr = param_group['lr']
                break
            print(
                f'Epoch:{context.epoch}\t'
                f'Iter:[{i}|{len(context.train_loader)}]\t'
                f'Lr:{current_lr:.5f}\t'
                f'Time:{avg_batch_time.avg:.5f}\t'
                f'Loss:{avg_losses.avg:.5f}\t'
                f'Accuracy:{avg_acc.avg:.5f}',
                flush=True,
            )

    if context.scheduler:
        context.scheduler.step()


def validate(model, context: DLContext) -> float:
    """Retrieves the accuracy the model via validation

    Args:
        model: The model to be validated
        context (DLContext): The context object

    Returns:
        float: Accuracy of the model
    """

    model.to(device=context.device)
    model.eval()

    avg_batch_time = AverageMeter()
    avg_acc = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (image, label) in enumerate(context.val_loader):
            image = image.to(context.device)
            label = label.to(context.device)

            output = model(image)
            avg_acc.update(compute_accuracy(output, label), image.size(0))

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print(
                    f'Test: [{i}/{len(context.val_loader)}]\tTime {avg_batch_time.avg:.5f}\tAcc@1 {avg_acc.avg:.5f}\t'
                )

        print(f' * Acc@1 {avg_acc.avg:.3f}')
    return avg_acc.avg


def calibrate(model, context: DLContext, eval=True):
    """Calibrates the fake-quantized model

    Args:
        model: The model to be validated
        context (DLContext): The context object
        eval: Flag to set train mode when used to do BN restore
    """
    model.to(device=context.device)
    if eval:
        model.eval()
    else:
        model.train()

    avg_batch_time = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (image, _) in enumerate(context.train_loader):

            if context.max_iteration is not None and i >= context.max_iteration:
                break

            image = image.to(device=context.device)

            model(image)

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(f'Calibrate: [{i}/{len(context.train_loader)}]\tTime {avg_batch_time.avg:.5f}\t')

            context.iteration += 1
