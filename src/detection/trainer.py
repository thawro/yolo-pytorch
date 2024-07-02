from src.base.trainer import Trainer

from .datamodule import DetectionDataModule
from .module import BaseDetectionModule
from .validator import CocoDetectionValidator


class DetectionTrainer(Trainer):
    datamodule: DetectionDataModule
    module: BaseDetectionModule
    validator: CocoDetectionValidator

    def get_validator(self) -> CocoDetectionValidator:
        preds_filepath = self.logger.log_path / "predictions.json"
        gt_annot_filepath = self.datamodule.val_ds.gt_annot_filepath
        return CocoDetectionValidator(
            gt_annot_filepath=gt_annot_filepath, preds_filepath=str(preds_filepath)
        )
