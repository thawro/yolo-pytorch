import math
from typing import Literal, Type

import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler

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

_mode = Literal["linear", "cos", "constant"]


class WarmupLR(_LRScheduler):
    def __init__(
        self, scheduler, optim_warmups: list[dict[str, float]], num_warmup=1, mode="linear"
    ):
        modes = ["linear", "cos", "constant"]
        if mode not in modes:
            raise ValueError(f"Expect warmup_strategy to be one of {modes} but got {mode}")
        self._scheduler = scheduler
        self.optim_warmups = optim_warmups
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate
        self._mode = mode
        if mode == "cos":
            self._warmup_func = self._warmup_cos
        elif mode == "linear":
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if (key != "optimizer" and key != "_scheduler")
        }
        wrapped_state_dict = {
            key: value for key, value in self._scheduler.__dict__.items() if key != "optimizer"
        }
        return {"wrapped": wrapped_state_dict, "wrapper": wrapper_state_dict}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict["wrapper"])
        self._scheduler.__dict__.update(state_dict["wrapped"])

    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for i, group in enumerate(self._scheduler.optimizer.param_groups):
            warmup_params = self.optim_warmups[i]
            print(warmup_params)

            group["warmup_start_lr"] = warmup_params["warmup_start_lr"]
            # min(self.init_lr, group["lr"])
            group["warmup_end_lr"] = warmup_params["warmup_end_lr"]
            if "momentum" in group:
                group["warmup_start_momentum"] = warmup_params["warmup_start_momentum"]
                group["warmup_end_momentum"] = warmup_params["warmup_end_momentum"]

    def _warmup_cos(self, start: int, end: int, pct: float) -> float:
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start: int, end: int, pct: float) -> float:
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start: int, end: int, pct: float) -> float:
        return (end - start) * pct + start

    def get_lr(self):
        if self._step_count >= self._num_warmup:
            return self._scheduler.get_lr()

        # warm up learning rate
        lrs = []
        momentums = []
        for group in self._scheduler.optimizer.param_groups:
            pct = self._step_count / (self._num_warmup - 1)
            warmup_lr = self._warmup_func(group["warmup_start_lr"], group["warmup_end_lr"], pct)
            lrs.append(warmup_lr)
            warmup_momentum = None
            if "momentum" in group:
                warmup_momentum = self._warmup_func(
                    group["warmup_start_momentum"], group["warmup_end_momentum"], pct
                )
            momentums.append(warmup_momentum)
        return lrs, momentums

    def step(self, *args):
        if self._step_count < self._num_warmup:
            lrs, momentums = self.get_lr()
            param_groups = self._scheduler.optimizer.param_groups
            for param_group, lr, momentum in zip(param_groups, lrs, momentums):
                param_group["lr"] = lr
                if "momentum" in param_group and momentum is not None:
                    param_group["momentum"] = momentum
            self._step_count += 1
        else:
            self._scheduler.step(*args)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


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
    # bias group is first
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
    lr_scheduler = LRSchedulerClass(optimizer, **params)
    return LRScheduler(lr_scheduler, interval=interval)
