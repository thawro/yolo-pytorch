import torch
from torch import Tensor, nn

from src.detection.architectures.blocks import DFL, Conv, DWConv, PWConv
from src.detection.architectures.yolo_v10.utils import _channels_dims


class YOLOv10ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            DWConv(in_channels, in_channels, 3),
            PWConv(in_channels, mid_channels),
            DWConv(mid_channels, mid_channels, 3),
            PWConv(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, P: Tensor) -> Tensor:
        return self.net(P)


class YOLOv10RegressionHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_reg_preds: int = 16):
        super().__init__()
        num_coords = 4
        num_reg_coords_preds = num_coords * num_reg_preds
        self.net = nn.Sequential(
            Conv(in_channels, mid_channels, 3),
            Conv(mid_channels, mid_channels, 3),
            nn.Conv2d(mid_channels, num_reg_coords_preds, 1),
        )

    def forward(self, P: Tensor) -> Tensor:
        return self.net(P)


class YOLOv10SingleScaleHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cls_mid_chanels: int,
        reg_mid_channels: int,
        num_classes: int,
        num_reg_preds: int = 1,
    ):
        super().__init__()
        self.reg_head = YOLOv10RegressionHead(in_channels, reg_mid_channels, num_reg_preds)
        self.cls_head = YOLOv10ClassificationHead(in_channels, cls_mid_chanels, num_classes)

    def forward(self, P: Tensor) -> Tensor:
        cls_preds = self.cls_head(P)
        reg_preds = self.reg_head(P)
        return torch.cat([reg_preds, cls_preds], dim=1)


class YOLOv10Head(nn.Module):
    def __init__(self, channels: _channels_dims, num_classes: int, num_reg_preds: int = 16):
        super().__init__()
        ch1, ch2, ch3, ch4, ch5 = channels
        reg_mid_ch = max(16, ch3 // 4, num_reg_preds * 4)
        cls_mid_ch = max(ch3, min(num_classes, 100))
        self.num_reg_preds = num_reg_preds
        self.num_classes = num_classes
        self.num_outputs = num_classes + num_reg_preds * 4
        self.P3_head = YOLOv10SingleScaleHead(
            ch3, cls_mid_ch, reg_mid_ch, num_classes, num_reg_preds
        )
        self.P4_head = YOLOv10SingleScaleHead(
            ch4, cls_mid_ch, reg_mid_ch, num_classes, num_reg_preds
        )
        self.P5_head = YOLOv10SingleScaleHead(
            ch5, cls_mid_ch, reg_mid_ch, num_classes, num_reg_preds
        )
        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.dfl = DFL(self.num_reg_preds) if self.num_reg_preds > 1 else nn.Identity()

    def forward(self, P3: Tensor, P4: Tensor, P5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3_preds = self.P3_head(P3)
        p4_preds = self.P4_head(P4)
        p5_preds = self.P5_head(P5)
        return p3_preds, p4_preds, p5_preds
