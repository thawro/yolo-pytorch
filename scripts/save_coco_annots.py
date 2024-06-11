from src.detection.config import DetectionConfig
from src.detection.datasets.coco import CocoDetectionDataset
from src.logger.pylogger import log
from src.utils.config import YAML_EXP_PATH

if __name__ == "__main__":
    log.info("Saving COCO keypoints annotations and crowd masks to files")
    cfg_path = YAML_EXP_PATH / "detection" / "yolo_v10_s.yaml"
    cfg_dict = DetectionConfig.from_yaml_to_dict(cfg_path)

    log.info("Started train split saving")
    train_ds = CocoDetectionDataset(**cfg_dict["dataloader"]["train_ds"], transform=None)
    train_ds.save_annots_to_files()
    log.info("Ended train split saving")

    log.info("Started val split saving")
    val_ds = CocoDetectionDataset(**cfg_dict["dataloader"]["val_ds"], transform=None)
    val_ds.save_annots_to_files()
    log.info("Ended val split saving")
