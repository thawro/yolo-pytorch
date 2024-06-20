"""Implementation of detection Module"""

import torch
from torch import Tensor

from src.base.module import BaseModule
from src.logger.loggers import log

from .datamodule import DetectionDataModule
from .loss import v8DetectionLoss, v10DetectLoss
from .model import BaseDetectionModel, v8DetectionModel, v10DetectionModel
from .results import DetectionResult

_batch = dict[str, Tensor]


class BaseDetectionModule(BaseModule):
    model: BaseDetectionModel
    loss_fn: v10DetectLoss | v8DetectionLoss
    datamodule: DetectionDataModule

    def __init__(
        self,
        model: v8DetectionModel | v10DetectionModel,
        loss_fn: v8DetectionLoss | v10DetectLoss,
        optimizers: dict,
        lr_schedulers: dict,
        multiscale: bool,
    ):
        super().__init__(model, loss_fn, optimizers, lr_schedulers, multiscale)
        log.info("..Using torch autocast to float16 in modules forward implementation..")

    def batch_to_device(self, batch: _batch) -> _batch:
        for k, v in batch.items():
            if isinstance(v, Tensor):
                batch[k] = v.to(
                    self.device, non_blocking=k == "images" and self.device != "cpu"
                )  # images
        return batch

    def _unwrap_batch(self, batch: _batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        images = batch["images"]
        gt_boxes_xywh = batch["boxes_xywh"]
        gt_classes = batch["classes"]
        gt_batch_idxs = batch["batch_idxs"]
        return images, gt_boxes_xywh, gt_classes, gt_batch_idxs

    def _optimizer_step(self, loss: Tensor):
        scaler, optimizer = self.scalers["optim"], self.optimizers["optim"]
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients

        scaler.step(optimizer)
        scaler.update()

    @staticmethod
    def create_results(images: Tensor, boxes_preds: list[Tensor]) -> list[DetectionResult]:
        images = images.detach().cpu()
        results = []
        for i in range(len(images)):
            result = DetectionResult(
                model_input_image=images[i],
                preds=boxes_preds[i].detach(),
            )
            results.append(result)
        return results


class v10DetectionModule(BaseDetectionModule):
    model: v10DetectionModel
    loss_fn: v10DetectLoss

    def training_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, gt_boxes_xywh, gt_classes, gt_batch_idxs = self._unwrap_batch(batch)
        if self.multiscale:
            images = self.random_images_rescale(images, min_scale=0.5, max_scale=1.5)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            one2one_preds, one2many_preds = self.model.train_forward(images)
            loss, losses_dict = self.loss_fn.calculate_loss(
                one2one_preds, one2many_preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
            )
        self._optimizer_step(loss)
        return {"loss": loss.item()} | losses_dict

    def validation_step(
        self, batch: _batch, batch_idx: int
    ) -> tuple[dict[str, float], list[DetectionResult]]:
        images, gt_boxes_xywh, gt_classes, gt_batch_idxs = self._unwrap_batch(batch)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            (one2one_preds, one2many_preds), boxes_preds = self.model.inference_forward(images)
            loss, losses_dict = self.loss_fn.calculate_loss(
                one2one_preds, one2many_preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
            )

        metrics = {"loss": loss.item()} | losses_dict
        results = self.create_results(images, boxes_preds)
        return metrics, results


class v8DetectionModule(BaseDetectionModule):
    model: v8DetectionModel
    loss_fn: v8DetectionLoss

    def training_step(self, batch: _batch, batch_idx: int) -> dict[str, float]:
        images, gt_boxes_xywh, gt_classes, gt_batch_idxs = self._unwrap_batch(batch)
        if self.multiscale:
            images = self.random_images_rescale(images, min_scale=0.5, max_scale=1.5)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            preds = self.model.train_forward(images)
            loss, losses_dict = self.loss_fn.calculate_loss(
                preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
            )
        self._optimizer_step(loss)
        return {"loss": loss.item()} | losses_dict

    def validation_step(
        self, batch: _batch, batch_idx: int
    ) -> tuple[dict[str, float], list[DetectionResult]]:
        images, gt_boxes_xywh, gt_classes, gt_batch_idxs = self._unwrap_batch(batch)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            preds, boxes_preds = self.model.inference_forward(images)
            loss, losses_dict = self.loss_fn.calculate_loss(
                preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
            )
        metrics = {"loss": loss.item()} | losses_dict
        results = self.create_results(images, boxes_preds)
        return metrics, results
