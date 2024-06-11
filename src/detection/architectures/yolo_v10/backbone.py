from torch import Tensor, nn

from src.detection.architectures.blocks import PSA, SPPF, C2f, C2fCiB, Conv, SCDown
from src.detection.architectures.yolo_v10.utils import _channels_dims, _num_backbone_bottlenecks


class YOLOv10Backbone(nn.Module):
    stage_1: nn.Module
    stage_2: nn.Module
    stage_3: nn.Module
    stage_4: nn.Module
    stage_5: nn.Module

    def __init__(self, channels: _channels_dims, num_bottlenecks: _num_backbone_bottlenecks):
        super().__init__()
        self.channels = channels
        self.num_bottlenecks = num_bottlenecks
        ch1, ch2, ch3, ch4, ch5 = channels
        nb2, nb3, nb4, nb5 = num_bottlenecks
        self.stage_1 = Conv(3, ch1, 3, 2)
        self.stage_2 = nn.Sequential(Conv(ch1, ch2, 3, 2), C2f(ch2, ch2, nb2, shortcut=True))
        self.stage_3 = nn.Sequential(Conv(ch2, ch3, 3, 2), C2f(ch3, ch3, nb3, shortcut=True))
        # stage_4 and stage_5 are set in subclasses for specific model versions (N/S/M/L/B/X)
        self.sppf = SPPF(ch5, ch5, 5)
        self.psa = PSA(ch5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        P1 = self.stage_1(x)
        P2 = self.stage_2(P1)
        P3 = self.stage_3(P2)
        P4 = self.stage_4(P3)
        P5 = self.stage_5(P4)
        P5 = self.sppf(P5)
        P5 = self.psa(P5)
        return P3, P4, P5


class YOLOv10Backbone_N(YOLOv10Backbone):
    def __init__(self, channels: _channels_dims, num_bottlenecks: _num_backbone_bottlenecks):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels
        nb2, nb3, nb4, nb5 = num_bottlenecks
        self.stage_4 = nn.Sequential(SCDown(ch3, ch4, 3, 2), C2f(ch4, ch4, nb4, shortcut=True))
        self.stage_5 = nn.Sequential(SCDown(ch4, ch5, 3, 2), C2f(ch5, ch5, nb5, shortcut=True))


class YOLOv10Backbone_S(YOLOv10Backbone):
    def __init__(self, channels: _channels_dims, num_bottlenecks: _num_backbone_bottlenecks):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels
        nb2, nb3, nb4, nb5 = num_bottlenecks
        self.stage_4 = nn.Sequential(SCDown(ch3, ch4, 3, 2), C2f(ch4, ch4, nb4, shortcut=True))
        self.stage_5 = nn.Sequential(
            SCDown(ch4, ch5, 3, 2), C2fCiB(ch5, ch5, nb5, shortcut=True, large_kernel=True)
        )


class YOLOv10Backbone_M(YOLOv10Backbone):
    def __init__(self, channels: _channels_dims, num_bottlenecks: _num_backbone_bottlenecks):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels
        nb2, nb3, nb4, nb5 = num_bottlenecks
        self.stage_4 = nn.Sequential(SCDown(ch3, ch4, 3, 2), C2f(ch4, ch4, nb4, shortcut=True))
        self.stage_5 = nn.Sequential(
            SCDown(ch4, ch5, 3, 2), C2fCiB(ch5, ch5, nb5, shortcut=True, large_kernel=False)
        )


YOLOv10Backbone_B = YOLOv10Backbone_L = YOLOv10Backbone_M


class YOLOv10Backbone_X(YOLOv10Backbone):
    def __init__(self, channels: _channels_dims, num_bottlenecks: _num_backbone_bottlenecks):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels
        nb2, nb3, nb4, nb5 = num_bottlenecks
        self.stage_4 = nn.Sequential(SCDown(ch3, ch4, 3, 2), C2fCiB(ch4, ch4, nb4, shortcut=True))
        self.stage_5 = nn.Sequential(
            SCDown(ch4, ch5, 3, 2), C2fCiB(ch5, ch5, nb5, shortcut=True, large_kernel=False)
        )
