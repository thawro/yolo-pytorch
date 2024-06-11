from src.base.trainer import Trainer

from .datamodule import DetectionDataModule
from .module import DetectionModule


class DetectionTrainer(Trainer):
    datamodule: DetectionDataModule
    module: DetectionModule
