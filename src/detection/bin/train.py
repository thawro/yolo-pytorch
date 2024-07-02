"""Train the keypoints estimation model"""

from src.base.bin.train import train
from src.detection.config import DetectionConfig
from src.utils.config import YAML_EXP_PATH


def main() -> None:
    cfg_path = YAML_EXP_PATH / "detection" / "yolo_v10_s.yaml"
    train(cfg_path, DetectionConfig)


if __name__ == "__main__":
    main()
