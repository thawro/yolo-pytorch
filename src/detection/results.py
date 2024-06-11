from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from src.base.results import BaseResult
from src.base.transforms import affine_transform, get_affine_transform
from src.utils.image import make_grid, match_size_to_src, stack_horizontally


class BaseDetectionResult(BaseResult):
    pass


class DetectionResult(BaseDetectionResult):
    preds: np.ndarray

    def __init__(
        self,
        model_input_image: Tensor,
        preds: list[Tensor],
        det_thr: float = 0.05,
    ):
        self.det_thr = det_thr

    def set_preds(self):
        self.preds = None

    def plot(self) -> dict[str, np.ndarray]:
        return {"detections": None}
