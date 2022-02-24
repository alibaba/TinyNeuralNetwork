import torch
import torch.distributed as dist


def is_distributed(args: dict) -> bool:
    """Whether DDP is activated

    Args:
        args (dict): The arguments passed in

    Returns:
        bool: Whether DDP is activated
    """

    return args.local_rank != -1


def is_main_process(args: dict):
    """Whether the process is running as the main process

    Args:
        args (dict): The arguments passed in

    Returns:
        bool: Whether the process is running as the main process
    """

    return not is_distributed(args) or args.local_rank == 0


def init_distributed(args: dict):
    """Initializes the DDP process pool and sets the default device

    Args:
        args (dict): The arguments passed in
    """

    if is_distributed(args):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)


def get_device(args: dict) -> torch.device:
    """Gets the default device

    Args:
        args (dict): The arguments passed in

    Returns:
        torch.device: The default device
    """

    if is_distributed(args):
        device = torch.device("cuda", args.local_rank)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
    return device


def deinit_distributed(args: dict):
    """Cleanups ddp process group

    Args:
        args (dict): The arguments passed in
    """

    if is_distributed(args):
        dist.destroy_process_group()
