import os
import typing

import torch

from .util import get_logger

log = get_logger(__name__, 'INFO')


class DLContext(object):
    def __init__(
        self,
        train_loader=None,
        train_sampler=None,
        val_loader=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        iter_scheduler=None,
        warmup_scheduler=None,
        epoch=0,
        max_epoch=None,
        iteration=0,
        warmup_iteration=0,
        max_iteration=None,
        gpu=None,
        device=None,
        grad_scaler=None,
        print_freq=50,
        train_func=None,
        validate_fun=None,
        custom_args: dict = None,
    ):
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iter_scheduler = iter_scheduler
        self.warmup_scheduler = warmup_scheduler
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.iteration = iteration
        self.warmup_iteration = warmup_iteration
        self.max_iteration = max_iteration
        self.gpu = gpu
        self.device = device
        self.grad_scaler = grad_scaler
        self.print_freq = print_freq
        self.best_acc = 0.0
        self.best_epoch = -1
        self.train_func = train_func
        self.validate_func = validate_fun
        self.custom_args = custom_args


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    model,
    context: DLContext,
    train_func: typing.Callable[[torch.nn.Module, DLContext], None],
    validate_func: typing.Callable[[torch.nn.Module, DLContext], float],
    distributed: bool = False,
    main_process: bool = True,
    qat: bool = False,
    work_dir: str = None,
):
    """The main function for the whole train process

    Args:
        model: The model to be trained
        context (DLContext): the DLContext object
        train_func (typing.Callable[[torch.nn.Module, DLContext], None]): The function to train the model by one step
        validate_func (typing.Callable[[torch.nn.Module, DLContext], float]): The function to get the \
        accuracy of the model
        distributed (bool, optional): Whether DDP training is used. Defaults to False.
        main_process (bool, optional): Whether the code runs in the main process. Defaults to True.
        qat (bool, optional): Whether to perform quantization-aware training. Defaults to False.
        work_dir (str, optional): Working directory. Defaults to None, which means "out".
    """

    if work_dir is None:
        work_dir = 'out'

    os.makedirs(work_dir, exist_ok=True)

    if isinstance(context.criterion, torch.nn.Module):
        context.criterion = context.criterion.to(device=context.device)

    best_acc = 0
    for i in range(context.max_epoch):
        context.epoch = i
        # calling `set_epoch` is required in distributd training
        if distributed:
            context.train_loader.sampler.set_epoch(i)
        train_func(model, context)

        # qat specific
        if qat:
            if context.epoch == context.max_epoch // 3:
                log.info("freeze quantizer parameters")
                model.apply(torch.quantization.disable_observer)
            elif context.epoch == context.max_epoch // 3 * 2:
                log.info("freeze batch norm mean and variance estimates")
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # only validate and save model in the main process
        if main_process:
            if distributed:
                # According to https://github.com/pytorch/pytorch/issues/54059, when validating via DDP,
                # it needs to be done on the original module.
                acc = validate_func(model.module, context)
            else:
                acc = validate_func(model, context)

            if qat:
                if context.epoch == context.max_epoch - 1:
                    save_path = os.path.join(work_dir, 'qat_last_model.pth')
                    torch.save(model.state_dict(), save_path)
            else:
                if acc > best_acc:
                    best_acc = acc
                    log.info(f"Best acc: {best_acc}")
                    save_path = os.path.join(work_dir, 'best_model.pth')
                    torch.save(model, save_path)

    # only validate the final model in the main process
    if main_process:
        if not qat:
            load_path = os.path.join(work_dir, 'best_model.pth')
            model.load_state_dict(torch.load(load_path).state_dict())
            validate_func(model, context)


def get_device() -> torch.device:
    """Gets the default device

    Returns:
        [torch.device]: The default device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def get_module_device(module: torch.nn.Module) -> typing.Optional[torch.device]:
    """Gets the device of the module

    Args:
        module (torch.nn.Module): The given module

    Returns:
        typing.Optional[torch.device]: The device of the module
    """

    assert isinstance(module, torch.nn.Module)

    device = None
    try:
        first_param = next(module.parameters())
        device = first_param.device
    except StopIteration:
        pass

    return device
