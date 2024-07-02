from contextlib import contextmanager
from copy import deepcopy

import thop
import torch
import torch.distributed as dist
from torch import nn


def get_flops(model: nn.Module, imgsz: int | tuple[int, int] = 640) -> float:
    """Return models GFLOPs."""
    p = next(model.parameters())
    if not isinstance(imgsz, tuple):
        imgsz = (imgsz, imgsz)  # expand if int/float
    img = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
    macs, params = thop.profile(deepcopy(model), inputs=[img], verbose=False)
    flops = macs * 2
    return flops


def model_info(model: nn.Module, detailed=False, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    num_params = get_num_params(model)
    num_gradients = get_num_gradients(model)
    num_layers = len(list(model.modules()))
    info = ""
    if detailed:
        info += f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}\n"
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            info += "%5g %40s %9s %12g %20s %10.3g %10.3g %10s \n" % (
                i,
                name,
                p.requires_grad,
                p.numel(),
                list(p.shape),
                p.mean(),
                p.std(),
                p.dtype,
            )
    flops = get_flops(model, imgsz)
    giga_flops = flops / 1e9
    fused = " (fused)" if model.is_fused else ""
    name = f"{model.__class__.__name__}{fused}"
    layers = f"{num_layers} layers"
    params = f"{num_params} parameters"
    gradients = f"{num_gradients} gradients"
    flops = f"{giga_flops:.1f} GFLOPs" if giga_flops else ""
    info += f"\n{name}, summary: {layers}, {params}, {gradients}, {flops}\n"
    return info, num_layers, num_params, num_gradients, flops


def get_num_params(model: nn.Module) -> int:
    """Return the total number of models parameters."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model: nn.Module) -> int:
    """Return the number of models parameters which require grad."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def parse_checkpoint(ckpt: dict) -> dict:
    redundant_prefixes = ["module.", "_orig_mod.", "net."]
    for key in list(ckpt.keys()):
        renamed_key = str(key)
        for prefix in redundant_prefixes:
            renamed_key = renamed_key.replace(prefix, "")
        ckpt[renamed_key] = ckpt.pop(key)
    return ckpt


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = dist.is_available() and dist.is_initialized()
    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])
