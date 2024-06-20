import os
import random

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int):
    from src.logger.pylogger import log

    log.info(f"..Setting seed to {seed}..")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    rank = get_rank()
    return rank == 0


def get_device_and_id(accelerator: str, use_DDP: bool) -> tuple[str, int]:
    if accelerator == "gpu" and torch.cuda.is_available():
        if use_DDP and "LOCAL_RANK" in os.environ:
            device_id = int(os.environ["LOCAL_RANK"])
        else:
            device_id = 0
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
        device_id = 0
    return device, device_id
