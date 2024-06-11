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
from src.datasets.coco import CocoDetectionDataset
from src.logger.pylogger import log

from .architectures import YOLOv10
from .datamodule import DetectionDataModule
from .loss import DetectionLoss
from .model import DetectionModel
from .module import DetectionModule
from .trainer import DetectionTrainer
from .transforms import DetectionTransform


@dataclass
class DetectionTransformConfig(TransformConfig):
    hm_resolutions: list[float]
    max_rotation: int
    min_scale: float
    max_scale: float
    scale_type: Literal["short", "long"]
    max_translate: int


@dataclass
class DetectionDatasetConfig(DatasetConfig):
    out_size: int
    hm_resolutions: list[float]
    num_kpts: int
    max_num_people: int
    sigma: float
    mosaic_probability: float


@dataclass
class DetectionDataloaderConfig(DataloaderConfig):
    train_ds: DetectionDatasetConfig
    val_ds: DetectionDatasetConfig


@dataclass
class DetectionInferenceConfig(InferenceConfig):
    use_flip: bool
    det_thr: float
    tag_thr: float
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
        log.info("..Creating DetectionDataModule..")
        transform = DetectionTransform(**self.transform.to_dict())
        train_ds = CocoDetectionDataset(
            **self.dataloader.train_ds.to_dict(), transform=transform.train
        )
        val_ds = CocoDetectionDataset(
            **self.dataloader.val_ds.to_dict(), transform=transform.inference
        )
        return DetectionDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            batch_size=self.dataloader.batch_size,
            pin_memory=self.dataloader.pin_memory,
            num_workers=self.dataloader.num_workers,
            use_DDP=self.trainer.use_DDP,
        )

    @property
    def architectures(self) -> dict[str, Type[nn.Module]]:
        return {"YOLOv10": YOLOv10}

    def _create_model(self) -> DetectionModel | nn.Module:
        log.info("..Creating Model..")
        net = self.create_net()

        if self.setup.is_train:
            return DetectionModel(net)
        else:
            return net

    def create_module(self) -> DetectionModule:
        log.info("..Creating MPPEDetectionModule..")
        loss_fn = DetectionLoss()
        model = self._create_model()
        module = DetectionModule(
            model=model,
            loss_fn=loss_fn,
            optimizers=self.get_optimizers_params(),
            lr_schedulers=self.get_lr_schedulers_params(),
        )
        return module

    @property
    def TrainerClass(self) -> Type[Trainer]:
        return DetectionTrainer

    def create_callbacks(self) -> list[BaseCallback]:
        callbacks = super().create_callbacks()
        examples_callback = ResultsPlotterCallback("heatmaps")
        callbacks.insert(-1, examples_callback)  # make sure it is before ArtifactsLoggerCallback
        return callbacks

    def create_inference_model(self, device: str = "cuda:0") -> InferenceDetectionModel:
        net = self.create_net()
        model = InferenceDetectionModel(
            net,
            device=device,
            det_thr=self.inference.det_thr,
            tag_thr=self.inference.tag_thr,
            use_flip=self.inference.use_flip,
            input_size=self.inference.input_size,
            ckpt_path=self.setup.ckpt_path,
        )
        return model
