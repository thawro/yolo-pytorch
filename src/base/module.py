import math
import random
from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules.loss import _Loss

from src.logger.loggers import BaseLogger
from src.logger.pylogger import log
from src.utils.optim import create_lr_scheduler, create_optimizer

from .callbacks import Callbacks
from .datamodule import DataModule
from .lr_scheduler import LRScheduler
from .model import BaseModel
from .results import BaseResult

SPLITS = ["train", "val", "test"]


class BaseModule:
    device: str
    device_id: int
    model: BaseModel
    logger: BaseLogger
    datamodule: DataModule
    callbacks: "Callbacks"
    current_epoch: int
    current_step: int
    limit_batches: int
    multiscale: bool
    optimizers: dict[str, optim.Optimizer]
    schedulers: dict[str, LRScheduler]
    scalers: dict[str, GradScaler]
    half_precision: bool
    dtype: torch.dtype = torch.float32

    def __init__(
        self,
        model: BaseModel,
        loss_fn: _Loss,
        optimizers: dict[str, dict],
        lr_schedulers: dict[str, dict],
        multiscale: bool = False,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self._optimizers = optimizers
        self._lr_schedulers = lr_schedulers
        self.multiscale = multiscale
        self.current_epoch = 0
        self.current_step = 0

    def create_optimizers(
        self,
    ) -> tuple[dict[str, optim.Optimizer], dict[str, LRScheduler]]:
        optimizers = {}
        lr_schedulers = {}
        # optims_warmups = {}
        for name, optimizer in self._optimizers.items():
            optimizers[name] = create_optimizer(
                net=self.model.net, name=optimizer["name"], **optimizer["params"]
            )

        for name, lr_scheduler in self._lr_schedulers.items():
            lr_schedulers[name] = create_lr_scheduler(
                optimizer=optimizers[name],
                name=lr_scheduler["name"],
                # optim_warmups=optims_warmups[name],
                interval=lr_scheduler["interval"],
                **lr_scheduler["params"],
            )

        return optimizers, lr_schedulers

    def set_optimizers(self):
        self.optimizers, self.lr_schedulers = self.create_optimizers()
        self.scalers = {name: GradScaler() for name in self.optimizers}

    def batch_to_device(self, batch: tuple) -> tuple:
        raise NotImplementedError()

    def pass_attributes(
        self,
        device_id: int,
        device: str,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
        limit_batches: int,
        half_precision: bool,
    ):
        self.device_id = device_id
        self.device = device
        self.logger = logger
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.limit_batches = limit_batches
        self.total_batches = datamodule.total_batches
        self.half_precision = half_precision
        if half_precision:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        if limit_batches > 0:
            self.total_batches = {k: limit_batches for k in self.total_batches}

    def random_images_rescale(
        self, images: Tensor, min_scale: float = 0.5, max_scale: float = 1.5
    ) -> Tensor:
        h, w = images.shape[2:]
        size = max(h, w)
        stride = self.model.stride
        min_size, max_size = int(size * min_scale), int(size * max_scale + stride)
        new_size = random.randrange(min_size, max_size) // stride * stride
        scale = new_size / max(h, w)
        if scale != 1:
            new_h = math.ceil(h * scale / stride) * stride
            new_w = math.ceil(w * scale / stride) * stride
            return F.interpolate(images, size=[new_h, new_w], mode="bilinear", align_corners=False)
        return images

    def set_attributes(self, **attributes):
        for name, attr in attributes.items():
            setattr(self, name, attr)

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
            lr = optimizer.param_groups[0]["lr"]
            log.info(f'\tLoaded "{name}" Optimizer state (lr = {lr})')
        for name, lr_scheduler in self.lr_schedulers.items():
            lr_scheduler.load_state_dict(state_dict["lr_schedulers"][name])
            last_epoch = lr_scheduler.lr_scheduler.last_epoch
            log.info(f'\tLoaded "{name}" LR Scheduler state (last_epoch = {last_epoch})')
        for name, scaler in self.scalers.items():
            scaler.load_state_dict(state_dict["scalers"][name])
            log.info(f'\tLoaded "{name}" Scaler state')

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        lr_schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.lr_schedulers.items()
        }
        scalers_state = {name: scaler.state_dict() for name, scaler in self.scalers.items()}
        model_state = self.model.state_dict()

        module_state = {
            "model": model_state,
            "optimizers": optimizers_state,
            "lr_schedulers": lr_schedulers_state,
            "scalers": scalers_state,
        }
        return module_state

    def _training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        self.stage = "train"
        self.is_train = True
        batch = self.batch_to_device(batch)
        metrics = self.training_step(batch, batch_idx)
        for name, lr_scheduler in self.lr_schedulers.items():
            if lr_scheduler.is_step_interval:
                lr_scheduler.step()
        return metrics

    def _validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> tuple[dict[str, float], list[BaseResult]]:
        self.stage = stage
        self.is_train = False
        batch = self.batch_to_device(batch)
        metrics, results = self.validation_step(batch, batch_idx)
        return metrics, results

    def on_epoch_start(self) -> None:
        pass

    def on_epoch_end(self) -> None:
        for name, lr_scheduler in self.lr_schedulers.items():
            if lr_scheduler.is_epoch_interval:
                lr_scheduler.step()

    @abstractmethod
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        raise NotImplementedError()

    @abstractmethod
    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> tuple[dict[str, float], list[BaseResult]]:
        raise NotImplementedError()
