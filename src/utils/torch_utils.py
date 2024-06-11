from copy import deepcopy

import thop
import torch
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


def model_info(model: nn.Module, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    num_params = get_num_params(model)
    num_gradients = get_num_gradients(model)
    num_layers = len(list(model.modules()))
    if detailed:
        print(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    flops = get_flops(model, imgsz)
    giga_flops = flops / 1e9
    fused = " (fused)" if model.is_fused else ""
    name = f"{model.__class__.__name__}{fused}"
    layers = f"{num_layers} layers"
    params = f"{num_params} parameters"
    gradients = f"{num_gradients} gradients"
    flops = f"{giga_flops:.1f} GFLOPs" if giga_flops else ""
    print(f"{name}, summary: {layers}, {params}, {gradients}, {flops}")
    return num_layers, num_params, num_gradients, flops


def get_num_params(model: nn.Module) -> int:
    """Return the total number of models parameters."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model: nn.Module) -> int:
    """Return the number of models parameters which require grad."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)
