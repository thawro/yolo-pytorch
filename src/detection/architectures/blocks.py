import torch
from torch import Tensor, nn

from src.base.architectures.blocks import Bottleneck, Conv, DWConv, PWConv
from src.base.architectures.utils import _activation, _kernel_size, fuse_conv_and_bn


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) module from https://arxiv.org/pdf/1406.4729v4."""

    def __init__(self, in_channels: int, out_channels: int, kernels: list[int] = [5, 9, 13]):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv_1 = Conv(in_channels, mid_channels, 1, 1)
        self.max_pools = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels])
        concat_dim = mid_channels * (len(kernels) + 1)
        self.conv_2 = Conv(concat_dim, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.conv_1(x)
        pooled_feats = [feats] + [max_pool(feats) for max_pool in self.max_pools]
        pooled_feats = torch.cat(pooled_feats, dim=1)
        return self.conv_2(pooled_feats)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (SPPF) based on https://github.com/ultralytics/yolov3/blob/master/models/common.py#L284"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_pools: int = 3):
        super().__init__()
        self.num_pools = num_pools
        mid_channels = in_channels // 2  # hidden channels
        self.conv_1 = Conv(in_channels, mid_channels, 1, 1)
        concat_dim = mid_channels * (num_pools + 1)
        self.conv_2 = Conv(concat_dim, out_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.conv_1(x)
        pooled_feats = [feats]
        for _ in range(self.num_pools):
            pooled_feats.append(self.max_pool(pooled_feats[-1]))
        pooled_feats = torch.cat(pooled_feats, 1)
        return self.conv_2(pooled_feats)


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion_rate: float = 0.5,
    ):
        """Initialize CSP bottleneck layer with two convolutions."""
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.mid_channels = int(out_channels * expansion_rate)  # hidden channels
        self.conv_1 = Conv(in_channels, 2 * self.mid_channels, 1, 1)
        self.conv_2 = Conv(2 * self.mid_channels, out_channels, 1)
        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(
                    self.mid_channels,
                    self.mid_channels,
                    shortcut,
                    groups,
                    kernel_size=[(3, 3), (3, 3)],
                    expansion_rate=1.0,
                )
                for _ in range(num_bottlenecks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through C2 layer."""
        feats = self.conv_1(x)
        feats = list(feats.split((self.mid_channels, self.mid_channels), dim=1))
        feats[-1] = self.bottlenecks(feats[-1])
        feats = torch.cat(feats, dim=1)
        return self.conv_2(feats)


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion_rate: float = 0.5,
    ):
        """Initialize CSP bottleneck layer with two convolutions."""
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.mid_channels = int(out_channels * expansion_rate)  # hidden channels
        self.conv_1 = Conv(in_channels, 2 * self.mid_channels, 1, 1)
        self.conv_2 = Conv((2 + num_bottlenecks) * self.mid_channels, out_channels, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(
                self.mid_channels,
                self.mid_channels,
                shortcut,
                groups,
                kernel_size=[(3, 3), (3, 3)],
                expansion_rate=1.0,
            )
            for _ in range(num_bottlenecks)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through C2f layer."""
        feats = self.conv_1(x)
        feats = list(feats.split((self.mid_channels, self.mid_channels), dim=1))
        for i in range(self.num_bottlenecks):
            feats.append(self.bottlenecks[i](feats[-1]))
        feats = torch.cat(feats, dim=1)
        return self.conv_2(feats)


class RepVGGDW(nn.Module):
    """Reparametrization Depthwise convolution based on https://arxiv.org/pdf/2101.03697.

    It is used as the second DW convolution in CiB block from YOLOv10.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = DWConv(channels, channels, kernel_size=7, stride=1, padding=3, activation=False)
        self.extra_conv = DWConv(
            channels, channels, kernel_size=3, stride=1, padding=1, activation=False
        )
        self.activation = nn.SiLU()
        self.is_fused = False

    def forward(self, x: Tensor) -> Tensor:
        if self.is_fused:
            return self.activation(self.conv(x))
        return self.activation(self.conv(x) + self.extra_conv(x))

    @torch.no_grad()
    def fuse(self) -> None:
        """Fuse two DW convolutions (along with BatchNorm) into single convolution."""
        if self.is_fused:
            return
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        extra_conv = fuse_conv_and_bn(self.extra_conv.conv, self.extra_conv.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        extra_conv_w = extra_conv.weight
        extra_conv_b = extra_conv.bias

        extra_conv_w = torch.nn.functional.pad(extra_conv_w, [2, 2, 2, 2])

        final_conv_w = conv_w + extra_conv_w
        final_conv_b = conv_b + extra_conv_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        self.conv = conv
        del self.extra_conv
        self.is_fused = True


class CiB(nn.Module):
    """CiB Block (Fig. 3a) from https://arxiv.org/pdf/2405.14458v1.

    if large_kernel is True, then 7x7 DWConv + 3x3 DWConv are used
    7x7 because: https://arxiv.org/pdf/2201.03545
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion_rate: float = 0.5,
        large_kernel: bool = False,
    ):
        super().__init__()
        mid_channels = 2 * int(out_channels * expansion_rate)  # hidden channels
        # 3×3 DW  -> ch_in
        # 1×1 PW  -> ch_mid
        # 3×3 DW  -> ch_mid   or 7×7 DW + 3×3 DW -> ch_mid
        # 1×1 PW  -> ch_out
        # 3×3 DW  -> ch_out
        if large_kernel:
            second_dw_conv = RepVGGDW(mid_channels)
        else:
            second_dw_conv = DWConv(mid_channels, mid_channels, kernel_size=3)

        self.cib = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            PWConv(in_channels, mid_channels),
            second_dw_conv,
            PWConv(mid_channels, out_channels),
            DWConv(out_channels, out_channels, kernel_size=3),
        )

        self.use_identity = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        if self.use_identity:
            return x + self.cib(x)
        return self.cib(x)


class C2fCiB(C2f):
    """C2f implementation with CiB blocks as bottlenecks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        large_kernel: bool = False,
        expansion_rate: float = 0.5,
    ):
        """Initialize C2f with CiB bottlenecks."""
        super().__init__(
            in_channels, out_channels, num_bottlenecks, shortcut, groups, expansion_rate
        )
        self.bottlenecks = nn.ModuleList(
            CiB(
                self.mid_channels,
                self.mid_channels,
                shortcut,
                expansion_rate=1.0,
                large_kernel=large_kernel,
            )
            for _ in range(num_bottlenecks)
        )


class DWSepConv(nn.Module):
    """Depthwise Separable Convolution from http://arxiv.org/pdf/1610.02357v3"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size,
        stride: int = 1,
        activation: _activation = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.dw_conv = DWConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )
        self.pw_conv = PWConv(
            out_channels, out_channels, activation=activation, use_batchnorm=use_batchnorm
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pw_conv(self.dw_conv(x))


class SCDown(nn.Module):
    """Spatial-channel decoupled downsampling layer from YOLOv10: https://arxiv.org/pdf/2405.14458v1"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size,
        stride: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.pw_conv = PWConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        self.dw_conv = DWConv(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            activation=False,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.dw_conv(self.pw_conv(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, activation=False)
        self.proj = Conv(dim, dim, 1, activation=False)
        self.pe = Conv(dim, dim, 3, 1, groups=dim, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):
    """Partial Self-Attention module from YOLOv10: https://arxiv.org/pdf/2405.14458v1.

    1. Apply 1x1 convolution
    2. Partition the features across channels into two parts
    3. Only feed one part into the NPSA blocks comprised of multi-head self-attention module (MHSA) and feed-forward network (FFN)
    4. Two parts are then concatenated and fused by a 1x1 convolution
    """

    def __init__(self, channels: int, expansion_rate: float = 0.5):
        super().__init__()
        self.mid_channels = int(channels * expansion_rate)
        self.conv_1 = Conv(channels, 2 * self.mid_channels, 1, 1)
        self.conv_2 = Conv(2 * self.mid_channels, channels, 1)

        self.attn = Attention(self.mid_channels, attn_ratio=0.5, num_heads=self.mid_channels // 64)
        self.ffn = nn.Sequential(
            Conv(self.mid_channels, self.mid_channels * 2, 1),
            Conv(self.mid_channels * 2, self.mid_channels, 1, activation=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.conv_1(x).split((self.mid_channels, self.mid_channels), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.conv_2(torch.cat((a, b), 1))


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL) from https://arxiv.org/pdf/2006.04388."""

    def __init__(self, num_reg_preds: int = 16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(num_reg_preds, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(num_reg_preds, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, num_reg_preds, 1, 1))
        self.num_reg_preds = num_reg_preds

    def forward(self, x: Tensor) -> Tensor:
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape
        # batch, num_reg_preds * 4, anchors -> batch, num_reg_preds, 4, anchors
        x = x.view(b, 4, self.num_reg_preds, a).transpose(2, 1).softmax(1)
        return self.conv(x).view(b, 4, a)
