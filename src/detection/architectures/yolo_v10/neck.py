from torch import Tensor, nn

from src.base.architectures.blocks import Concat, Conv
from src.detection.architectures.blocks import C2f, C2fCiB, SCDown
from src.detection.architectures.yolo_v10.utils import _channels_dims


class YOLOv10Neck(nn.Module):
    upsample: nn.Upsample
    concat: Concat
    conv_td_p4: C2f | C2fCiB
    conv_td_p3: C2f | C2fCiB
    downsample_p3: Conv
    conv_bu_p4: C2f | C2fCiB
    downsample_p4: SCDown
    conv_bu_p5: C2f | C2fCiB

    def __init__(self, channels: _channels_dims, num_bottlenecks: int):
        super().__init__()
        self.channels = channels
        self.num_bottlenecks = num_bottlenecks
        self.upsample = nn.Upsample(None, scale_factor=2, mode="nearest")
        self.concat = Concat(dim=1)

    def _bottom_up(self, P3: Tensor, P4: Tensor, P5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        P5_ups = self.upsample(P5)
        P5_P4 = self.concat([P5_ups, P4])
        P4 = self.conv_td_p4(P5_P4)

        P4_ups = self.upsample(P4)
        P4_P3 = self.concat([P4_ups, P3])
        P3 = self.conv_td_p3(P4_P3)
        return P3, P4, P5

    def _top_down(self, P3: Tensor, P4: Tensor, P5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        P3_downs = self.downsample_p3(P3)
        P3_P4 = self.concat([P3_downs, P4])
        P4 = self.conv_bu_p4(P3_P4)

        P4_downs = self.downsample_p4(P4)
        P4_P5 = self.concat([P4_downs, P5])
        P5 = self.conv_bu_p5(P4_P5)
        return P3, P4, P5

    def forward(self, P3: Tensor, P4: Tensor, P5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        P3, P4, P5 = self._bottom_up(P3, P4, P5)
        P3, P4, P5 = self._top_down(P3, P4, P5)
        return P3, P4, P5


class YOLOv10Neck_N(YOLOv10Neck):
    def __init__(self, channels: _channels_dims, num_bottlenecks: int):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels

        self.conv_td_p4 = C2f((ch5 + ch4), ch4, num_bottlenecks=num_bottlenecks)
        self.conv_td_p3 = C2f((ch4 + ch3), ch3, num_bottlenecks=num_bottlenecks)

        self.downsample_p3 = Conv(ch3, ch3, 3, 2)
        self.conv_bu_p4 = C2f((ch3 + ch4), ch4, num_bottlenecks=num_bottlenecks)
        self.downsample_p4 = SCDown(ch4, ch4, 3, 2)
        self.conv_bu_p5 = C2fCiB(
            (ch5 + ch4), ch5, shortcut=True, num_bottlenecks=num_bottlenecks, large_kernel=True
        )


YOLOv10Neck_S = YOLOv10Neck_N


class YOLOv10Neck_M(YOLOv10Neck):
    def __init__(self, channels: _channels_dims, num_bottlenecks: int):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels

        self.conv_td_p4 = C2f((ch5 + ch4), ch4, num_bottlenecks=num_bottlenecks)
        self.conv_td_p3 = C2f((ch4 + ch3), ch3, num_bottlenecks=num_bottlenecks)

        self.downsample_p3 = Conv(ch3, ch3, 3, 2)
        self.conv_bu_p4 = C2fCiB((ch3 + ch4), ch4, num_bottlenecks=num_bottlenecks, shortcut=True)
        self.downsample_p4 = SCDown(ch4, ch4, 3, 2)
        self.conv_bu_p5 = C2fCiB(
            (ch5 + ch4), ch5, shortcut=True, num_bottlenecks=num_bottlenecks, large_kernel=False
        )


class YOLOv10Neck_B(YOLOv10Neck):
    def __init__(self, channels: _channels_dims, num_bottlenecks: int):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels

        self.conv_td_p4 = C2fCiB((ch5 + ch4), ch4, num_bottlenecks=num_bottlenecks, shortcut=True)
        self.conv_td_p3 = C2f((ch4 + ch3), ch3, num_bottlenecks=num_bottlenecks)

        self.downsample_p3 = Conv(ch3, ch3, 3, 2)
        self.conv_bu_p4 = C2fCiB((ch3 + ch4), ch4, num_bottlenecks=num_bottlenecks, shortcut=True)
        self.downsample_p4 = SCDown(ch4, ch4, 3, 2)
        self.conv_bu_p5 = C2fCiB(
            (ch5 + ch4), ch5, shortcut=True, num_bottlenecks=num_bottlenecks, large_kernel=False
        )


class YOLOv10Neck_L(YOLOv10Neck):
    def __init__(self, channels: _channels_dims, num_bottlenecks: int):
        super().__init__(channels, num_bottlenecks)
        ch1, ch2, ch3, ch4, ch5 = channels

        self.conv_td_p4 = C2fCiB((ch5 + ch4), ch4, num_bottlenecks=num_bottlenecks, shortcut=True)
        self.conv_td_p3 = C2f((ch4 + ch3), ch3, num_bottlenecks=num_bottlenecks)

        self.downsample_p3 = Conv(ch3, ch3, 3, 2)
        self.conv_bu_p4 = C2fCiB((ch3 + ch4), ch4, num_bottlenecks=num_bottlenecks, shortcut=True)
        self.downsample_p4 = SCDown(ch4, ch4, 3, 2)
        self.conv_bu_p5 = C2fCiB(
            (ch5 + ch4), ch5, shortcut=True, num_bottlenecks=num_bottlenecks, large_kernel=False
        )


YOLOv10Neck_X = YOLOv10Neck_L
