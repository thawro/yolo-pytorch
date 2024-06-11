"""DataModule used to load DataLoaders"""

from src.base.datamodule import DataModule
from src.datasets.coco import CocoDataset


class DetectionDataModule(DataModule):
    train_ds: CocoDataset
    val_ds: CocoDataset
    test_ds: CocoDataset
