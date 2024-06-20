from typing import Literal, Type

from torch import nn, optim

from src.base.lr_scheduler import LRScheduler

_optimizers = Literal["Adam", "Adamax", "Adadelta", "Adagrad", "AdamW", "SGD", "RMSprop"]


optimizers: dict[str, Type[optim.Optimizer]] = {
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
}

_lr_schedulers = Literal[
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "MultiStepLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "ExponentialLR",
    "PolynomialLR",
]
lr_schedulers: dict[str, Type[optim.lr_scheduler.LRScheduler]] = {
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "PolynomialLR": optim.lr_scheduler.PolynomialLR,
}


def create_optimizer(net: nn.Module, name: _optimizers, **params) -> optim.Optimizer:
    OptimizerClass = optimizers[name]
    g_bias = []
    g_norm = []
    g_rest = []
    norm_layers = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for module_name, module in net.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                g_bias.append(param)
            elif isinstance(module, norm_layers):  # weight (no decay)
                g_norm.append(param)
            else:  # weight (with decay)
                g_rest.append(param)
    weight_decay = params.pop("weight_decay", 0)
    optimizer = OptimizerClass(g_bias, **params)
    optimizer.add_param_group({"params": g_rest, "weight_decay": weight_decay})  # group with decay
    optimizer.add_param_group({"params": g_norm, "weight_decay": 0.0})  # group with Norm weights
    return optimizer


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    name: _lr_schedulers,
    interval: Literal["epoch", "step"],
    **params,
) -> LRScheduler:
    LRSchedulerClass = lr_schedulers[name]
    return LRScheduler(LRSchedulerClass(optimizer, **params), interval=interval)
