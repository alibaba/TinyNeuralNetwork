import time
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tinynn.util.train_util import AverageMeter, DLContext


def get_dataloader(
    data_path: str,
    img_size: int = 224,
    batch_size: int = 128,
    worker: int = 4,
    distributed: bool = False,
    download: bool = False,
) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Constructs the dataloaders for training and validating

    Args:
        data_path (str): The path of the dataset
        img_size (int, optional): The size of the image. Defaults to 224.
        batch_size (int, optional): The batch size of the dataloader. Defaults to 128.
        worker (int, optional): The number of workers. Defaults to 4.
        distributed (bool, optional): Whether to use DDP. Defaults to False.
        download (bool, optional): Whether to download the dataset. Defaults to False.

    Returns:
        typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The dataloaders for training and \
            validating
    """

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=worker,
        pin_memory=True,
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True
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

        if context.max_iteration is not None and context.iteration >= context.max_iteration:
            break

        avg_data_time.update(time.time() - end)
        image = image.to(device=context.device)
        output = model(image)

        if isinstance(context.criterion, nn.BCEWithLogitsLoss):
            label.unsqueeze_(1)
            label_onehot = torch.FloatTensor(label.shape[0], 10)
            label_onehot.zero_()
            label_onehot.scatter_(1, label, 1)
            label.squeeze_(1)
            label_onehot = label_onehot.to(device=context.device)
            label = label.to(device=context.device)
            loss = context.criterion(output, label_onehot)
        else:
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
                f'Accuracy:{avg_acc.avg:.5f}'
            )

        if context.warmup_scheduler is not None and context.warmup_iteration > context.iteration:
            context.warmup_scheduler.step()

        context.iteration += 1

    if context.scheduler and context.warmup_iteration <= context.iteration:
        context.scheduler.step()


def train_one_epoch_distill(model, context: DLContext):
    """Train the model for one epoch with distilling

    Args:
        model: Student model
        context (DLContext): The context object
    """
    A = context.custom_args['distill_A']
    T = context.custom_args['distill_T']
    teacher = context.custom_args['distill_teacher']

    avg_batch_time = AverageMeter()
    avg_data_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_origin_losses = AverageMeter()
    avg_acc = AverageMeter()

    model.to(device=context.device)
    model.train()

    teacher.to(device=context.device)
    teacher.eval()
    end = time.time()
    for i, (image, label) in enumerate(context.train_loader):

        if context.max_iteration is not None and context.iteration >= context.max_iteration:
            break

        avg_data_time.update(time.time() - end)
        image = image.to(device=context.device)
        output = model(image)

        if isinstance(context.criterion, nn.BCEWithLogitsLoss):
            label.unsqueeze_(1)
            label_onehot = torch.FloatTensor(label.shape[0], 10)
            label_onehot.zero_()
            label_onehot.scatter_(1, label, 1)
            label.squeeze_(1)
            label_onehot = label_onehot.to(device=context.device)
            label = label.to(device=context.device)
            origin_loss = context.criterion(output, label_onehot)
        else:
            label = label.to(device=context.device)
            origin_loss = context.criterion(output, label)

        with torch.no_grad():
            label_teacher = teacher(image)

        distill_loss = (
            F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(label_teacher / T, dim=1), reduction='batchmean')
            * T
            * T
        )

        avg_origin_losses.update(origin_loss * (1 - A))
        loss = origin_loss * (1 - A) + distill_loss * A

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
                f'Lr:{current_lr:.6f}\t'
                f'Time:{avg_batch_time.val:.3f}\t'
                f'Loss:{avg_origin_losses.val:.6f} | {avg_losses.val - avg_origin_losses.val:.6f}\t'
                f'Accuracy:{avg_acc.val:.3f}'
            )

        if context.warmup_scheduler is not None and context.warmup_iteration > context.iteration:
            context.warmup_scheduler.step()

        context.iteration += 1

    if context.scheduler and context.warmup_iteration <= context.iteration:
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
            image = image.to(device=context.device)
            label = label.to(device=context.device)

            output = model(image)
            avg_acc.update(compute_accuracy(output, label), image.size(0))

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(
                    f'Test: [{i}/{len(context.val_loader)}]\tTime {avg_batch_time.avg:.5f}\tAcc@1 {avg_acc.avg:.5f}\t'
                )

        print(f'Validation Acc@1 {avg_acc.avg:.3f}')
    return avg_acc.avg


def calibrate(model, context: DLContext):
    """Calibrates the fake-quantized model

    Args:
        model: The model to be validated
        context (DLContext): The context object

    """

    model.to(device=context.device)
    model.eval()

    avg_batch_time = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (image, _) in enumerate(context.val_loader):

            if context.max_iteration is not None and i >= context.max_iteration:
                break

            image = image.to(device=context.device)

            model(image)

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(f'Calibrate: [{i}/{len(context.val_loader)}]\tTime {avg_batch_time.avg:.5f}\t')

            context.iteration += 1
