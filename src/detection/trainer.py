from src.base.trainer import Trainer

from .datamodule import DetectionDataModule
from .module import BaseDetectionModule


class DetectionTrainer(Trainer):
    datamodule: DetectionDataModule
    module: BaseDetectionModule
