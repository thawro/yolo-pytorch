import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from src.base.transforms import ComposeTransform

COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class ToTensor(object):
    def __call__(
        self, image: np.ndarray, boxes: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        return F.to_tensor(image), boxes


class Normalize(object):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, boxes: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes


class ComposeDetectionTransform(ComposeTransform):
    def __call__(self, image: np.ndarray, boxes: list) -> tuple[torch.Tensor | np.ndarray, list]:
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        return image, boxes


class DetectionTransform:
    def __init__(
        self,
        out_size: int,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.out_size = out_size
        self.mean = mean
        self.std = std
        self.train = ComposeDetectionTransform(
            [
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
        self.inference = ComposeDetectionTransform(
            [
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
