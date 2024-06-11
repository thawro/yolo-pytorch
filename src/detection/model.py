import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn

from src.base.model import BaseImageModel, BaseInferenceModel
from src.base.transforms import resize_align_multi_scale
from src.logger.pylogger import log


class DetectionModel(BaseImageModel):
    def __init__(self, net: nn.Module):
        super().__init__(net, ["images"], ["objects"])

    def init_weights(self):
        # TODO
        pass

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        return super().forward(images)

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 512, 512)


class InferenceDetectionModel(BaseInferenceModel):
    def __init__(
        self,
        net: nn.Module,
        input_size: int = 512,
        device: str = "cuda:0",
        ckpt_path: str | None = None,
    ):
        super().__init__(net, input_size, device, ckpt_path)
