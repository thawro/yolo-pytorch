"""Implementation of specialized Module"""

import torch
from torch import Tensor

from src.base.module import BaseModule
from src.logger.loggers import log

from .datamodule import DetectionDataModule
from .loss import DetectionLoss
from .model import DetectionModel
from .results import DetectionResult

_batch = tuple[Tensor, Tensor]


class DetectionModule(BaseModule):
    model: DetectionModel
    loss_fn: DetectionLoss
    datamodule: DetectionDataModule

    def __init__(
        self,
        model: DetectionModel,
        loss_fn: DetectionLoss,
        optimizers: dict,
        lr_schedulers: dict,
    ):
        super().__init__(model, loss_fn, optimizers, lr_schedulers)
        log.info("..Using torch autocast to float16 in modules forward implementation..")

    def batch_to_device(self, batch: _batch) -> _batch:
        images, boxes = batch
        images = images.cuda()
        boxes = boxes.cuda()
        return images, boxes

    def training_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, boxes = batch

        scaler, optimizer = self.scalers["optim"], self.optimizers["optim"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = self.model.net(images)
            loss_1, loss_2 = self.loss_fn.calculate_loss(preds, boxes)
        loss = loss_1 + loss_2
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics = {"loss": loss.item(), "loss_1": loss_1.item(), "loss_2": loss_2.item()}
        return metrics

    def validation_step(
        self, batch: _batch, batch_idx: int
    ) -> tuple[dict[str, float], list[DetectionResult]]:
        images, boxes = batch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = self.model.net(images)
            loss_1, loss_2 = self.loss_fn.calculate_loss(preds, boxes)
        loss = loss_1 + loss_2

        metrics = {"loss": loss.item(), "loss_1": loss_1.item(), "loss_2": loss_2.item()}

        preds = preds.detach()
        images = images.detach().cpu()
        results = []
        for i in range(len(images)):
            result = DetectionResult(
                model_input_image=images[i],
                preds=preds,
                det_thr=0.1,
            )
            results.append(result)
        return metrics, results
