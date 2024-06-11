from typing import Literal, Type

from torch import Tensor, nn

from src.base.architectures.blocks import Conv, DWConv, PWConv
from src.base.architectures.utils import make_divisible
from src.detection.architectures.blocks import RepVGGDW
from src.detection.architectures.yolo_v10.backbone import (
    YOLOv10Backbone,
    YOLOv10Backbone_B,
    YOLOv10Backbone_L,
    YOLOv10Backbone_M,
    YOLOv10Backbone_N,
    YOLOv10Backbone_S,
    YOLOv10Backbone_X,
)
from src.detection.architectures.yolo_v10.head import YOLOv10Head
from src.detection.architectures.yolo_v10.neck import (
    YOLOv10Neck,
    YOLOv10Neck_B,
    YOLOv10Neck_L,
    YOLOv10Neck_M,
    YOLOv10Neck_N,
    YOLOv10Neck_S,
    YOLOv10Neck_X,
)
from src.detection.architectures.yolo_v10.utils import _channels_dims, _num_backbone_bottlenecks

CHANNELS = 64, 128, 256, 512, 1024
NUM_BACKBONE_BOTTLENECKS = 3, 6, 6, 3
NUM_NECK_BOTTLENECKS = 3


class BaseYOLOv10(nn.Module):
    channels: _channels_dims
    num_backbone_bottlenecks: _num_backbone_bottlenecks
    num_neck_bottlenecks: int
    Backbone: Type[YOLOv10Backbone]
    Neck: Type[YOLOv10Neck]
    width: float
    max_channels: int
    depth: float

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self._set_num_channels_and_bottlenecks()

        self.backbone = self.Backbone(self.channels, self.num_backbone_bottlenecks)
        self.neck = self.Neck(self.channels, num_bottlenecks=self.num_neck_bottlenecks)
        self.head = YOLOv10Head(self.channels, num_classes, num_reg_preds=16)
        self.is_fused = False

    def _set_num_channels_and_bottlenecks(self):
        channels = []
        for ch in CHANNELS:
            channels.append(make_divisible(min(ch, self.max_channels) * self.width, 8))

        num_backbone_bottlenecks = []
        for nb in NUM_BACKBONE_BOTTLENECKS:
            num_backbone_bottlenecks.append(int(nb * self.depth))

        self.channels = tuple(channels)
        self.num_backbone_bottlenecks = tuple(num_backbone_bottlenecks)
        self.num_neck_bottlenecks = int(NUM_NECK_BOTTLENECKS * self.depth)

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        P3, P4, P5 = self.backbone(img)
        N3, N4, N5 = self.neck(P3, P4, P5)
        H3, H4, H5 = self.head(N3, N4, N5)
        return H3, H4, H5

    def fuse(self):
        if self.is_fused:
            return
        for name, module in self.named_modules():
            if isinstance(module, (RepVGGDW, Conv, DWConv, PWConv)):
                module.fuse()
        self.is_fused = True


class YOLOv10n(BaseYOLOv10):
    Backbone = YOLOv10Backbone_N
    Neck = YOLOv10Neck_N
    depth = 1 / 3
    width = 0.25
    max_channels = 1024


class YOLOv10s(BaseYOLOv10):
    Backbone = YOLOv10Backbone_S
    Neck = YOLOv10Neck_S
    depth = 1 / 3
    width = 0.5
    max_channels = 1024


class YOLOv10m(BaseYOLOv10):
    Backbone = YOLOv10Backbone_M
    Neck = YOLOv10Neck_M
    depth = 2 / 3
    width = 0.75
    max_channels = 768


class YOLOv10b(BaseYOLOv10):
    Backbone = YOLOv10Backbone_B
    Neck = YOLOv10Neck_B
    depth = 2 / 3
    width = 1.0
    max_channels = 512


class YOLOv10l(BaseYOLOv10):
    Backbone = YOLOv10Backbone_L
    Neck = YOLOv10Neck_L
    depth = 1.0
    width = 1.0
    max_channels = 512


class YOLOv10x(BaseYOLOv10):
    Backbone = YOLOv10Backbone_X
    Neck = YOLOv10Neck_X
    depth = 1.0
    width = 1.25
    max_channels = 512


_versions = Literal["N", "S", "M", "B", "L", "X"]

version2model = {
    "N": YOLOv10n,
    "S": YOLOv10s,
    "M": YOLOv10m,
    "B": YOLOv10b,
    "L": YOLOv10l,
    "X": YOLOv10x,
}


class YOLOv10(BaseYOLOv10):
    def __new__(cls, version: _versions, num_classes: int = 80) -> BaseYOLOv10:
        return version2model[version](num_classes)
