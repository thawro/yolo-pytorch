import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from src.base.transforms import ComposeTransform


class ToTensor(object):
    def __call__(
        self, image: np.ndarray, boxes_xywh: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        return F.to_tensor(image), boxes_xywh


class Normalize(object):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, boxes_xywh: list[np.ndarray]
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes_xywh


class ComposeDetectionTransform(ComposeTransform):
    def __call__(
        self, image: np.ndarray, boxes_xywh: list
    ) -> tuple[torch.Tensor | np.ndarray, list]:
        for transform in self.transforms:
            image, boxes_xywh = transform(image, boxes_xywh)
        return image, boxes_xywh


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
