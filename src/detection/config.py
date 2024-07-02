from dataclasses import dataclass
from typing import Literal, Type

from torch import nn

from src.base.callbacks import BaseCallback, ResultsPlotterCallback
from src.base.config import (
    BaseConfig,
    DataloaderConfig,
    DatasetConfig,
    InferenceConfig,
    TransformConfig,
)
from src.base.trainer import Trainer
from src.detection.datasets.coco import CocoDetectionDataset
from src.logger.pylogger import log_msg
from src.utils.torch_utils import torch_distributed_zero_first
from src.utils.training import get_rank

from .architectures import YOLOv8, YOLOv10
from .datamodule import DetectionDataModule
from .datasets.transforms import DetectionTransform
from .loss import v8DetectionLoss, v10DetectLoss
from .model import v8DetectionModel, v10DetectionModel
from .module import BaseDetectionModule, v8DetectionModule, v10DetectionModule
from .trainer import DetectionTrainer


@dataclass
class DetectionTransformConfig(TransformConfig):
    scale_min: float
    scale_max: float
    degrees: int
    translate: float
    flip_lr: float
    flip_ud: float


@dataclass
class DetectionDataloaderConfig(DataloaderConfig):
    train_ds: DatasetConfig
    val_ds: DatasetConfig


@dataclass
class DetectionInferenceConfig(InferenceConfig):
    use_flip: bool
    input_size: int


@dataclass
class DetectionConfig(BaseConfig):
    dataloader: DetectionDataloaderConfig
    transform: DetectionTransformConfig
    inference: DetectionInferenceConfig

    @property
    def is_debug(self) -> bool:
        return self.trainer.limit_batches > 0

    def create_datamodule(self) -> DetectionDataModule:
        log_msg("-> Creating DetectionDataModule")
        transform = DetectionTransform(**self.transform.to_dict())
        with torch_distributed_zero_first(get_rank()):  # init dataset *.cache only once if DDP
            train_ds = CocoDetectionDataset(
                **self.dataloader.train_ds.to_dict(), size=transform.size
            )
            train_ds.set_transform(transform.train_transform(train_ds))
            val_ds = CocoDetectionDataset(**self.dataloader.val_ds.to_dict(), size=transform.size)
            val_ds.set_transform(transform.inference_transform())
        size = transform.size
        return DetectionDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            input_size=size if isinstance(size, int) else size[0],
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            collate_fn=CocoDetectionDataset.collate_fn,
            use_DDP=self.trainer.use_DDP,
        )

    @property
    def architectures(self) -> dict[str, Type[nn.Module]]:
        return {"YOLOv10": YOLOv10, "YOLOv8": YOLOv8}

    @property
    def losses(self) -> dict[str, Type[v8DetectionLoss | v10DetectLoss]]:
        return {"YOLOv10": v10DetectLoss, "YOLOv8": v8DetectionLoss}

    @property
    def models(self) -> dict[str, Type[v10DetectionModel | v8DetectionModel]]:
        return {"YOLOv10": v10DetectionModel, "YOLOv8": v8DetectionModel}

    @property
    def modules(self) -> dict[str, Type[BaseDetectionModule]]:
        return {"YOLOv10": v10DetectionModule, "YOLOv8": v8DetectionModule}

    def _create_model(self) -> v10DetectionModel | v8DetectionModel | nn.Module:
        log_msg("-> Creating Model")
        net = self.create_net()
        if self.setup.is_train:
            ModelClass = self.models[self.setup.architecture]
            return ModelClass(net, stride=self.net.stride)
        else:
            return net

    def create_module(self) -> BaseDetectionModule:
        log_msg("-> Creating DetectionModule")
        model = self._create_model()
        LossClass = self.losses[self.setup.architecture]
        loss_fn = LossClass(strides=[8, 16, 32])
        ModuleClass = self.modules[self.setup.architecture]
        module = ModuleClass(
            model=model,
            loss_fn=loss_fn,
            optimizers=self.get_optimizers_params(),
            lr_schedulers=self.get_lr_schedulers_params(),
            multiscale=self.module.multiscale,
        )
        return module

    @property
    def TrainerClass(self) -> Type[Trainer]:
        return DetectionTrainer

    def create_callbacks(self) -> list[BaseCallback]:
        callbacks = super().create_callbacks()
        examples_callback = ResultsPlotterCallback("detections", ncols=8)
        callbacks.insert(-1, examples_callback)  # make sure it is before ArtifactsLoggerCallback
        return callbacks

    # def create_inference_model(self, device: str = "cuda:0") -> InferenceDetectionModel:
    #     net = self.create_net()
    #     model = InferenceDetectionModel(
    #         net,
    #         device=device,
    #         use_flip=self.inference.use_flip,
    #         input_size=self.inference.input_size,
    #         ckpt_path=self.setup.ckpt_path,
    #     )
    #     return model
