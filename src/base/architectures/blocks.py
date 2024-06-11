"""Helpers for architectures implementation."""

import math
from typing import Literal

import torch
from torch import Tensor, nn

from src.base.architectures.utils import _activation, _kernel_size, fuse_conv_and_bn


def autopad(
    kernel_size: _kernel_size, padding: int | None = None, dilation: int = 1
) -> int | list[int]:
    """Pad to 'same' shape outputs."""
    k, p, d = kernel_size, padding, dilation
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:  # auto padding
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_activation = nn.SiLU()  # default activation
    conv: nn.Conv2d
    bn: nn.BatchNorm2d | nn.Identity
    activation: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: _activation = True,
        use_batchnorm: bool = True,
    ):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.is_fused = False
        same_padding = autopad(kernel_size, padding, dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            same_padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        if activation is True:
            self.activation = self.default_activation
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Perform transposed convolution of 2D data."""
        if self.is_fused:
            return self.activation(self.conv(x))
        return self.activation(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if not isinstance(self.bn, nn.Identity) and not self.is_fused:
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
            del self.bn
            self.is_fused = True


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        activation: _activation = True,
        use_batchnorm: bool = True,
    ):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=math.gcd(in_channels, out_channels),
            dilation=dilation,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )


class PWConv(Conv):
    """Point-wise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: _activation = True,
        use_batchnorm: bool = True,
    ):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size = 1,
        activation: _activation = nn.ReLU(),
        use_batchnorm: bool = True,
    ):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.pw_conv = PWConv(
            in_channels, out_channels, activation=False, use_batchnorm=use_batchnorm
        )
        self.dw_conv = DWConv(
            out_channels,
            out_channels,
            kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2 convolutions to input tensor."""
        return self.dw_conv(self.pw_conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_size: list[_kernel_size] = [3, 3],
        expansion_rate: float = 0.5,
    ):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        mid_channels = int(out_channels * expansion_rate)  # hidden channels
        self.conv_1 = Conv(in_channels, mid_channels, kernel_size[0], 1)
        self.conv_2 = Conv(mid_channels, out_channels, kernel_size[1], 1, groups=groups)
        self.use_identity = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """'forward()' applies the YOLO FPN to input data."""
        if self.use_identity:
            return x + self.conv_2(self.conv_1(x))
        return self.conv_2(self.conv_1(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _kernel_size = 1,
        stride: int = 1,
        groups: int = 1,
        activation: _activation = True,
    ):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        mid_channels = out_channels // 2  # hidden channels
        padding = None
        self.conv_1 = Conv(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            groups,
            activation=activation,
        )
        self.conv_2 = Conv(
            mid_channels,
            mid_channels,
            5,
            1,
            padding,
            mid_channels,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        feats_1 = self.conv_1(x)
        feats_2 = self.conv_2(feats_1)
        combined_feats = torch.cat((feats_1, feats_2), 1)
        return combined_feats


class ChannelAttention(nn.Module):
    """Channel-attention module from http://arxiv.org/pdf/1807.06521."""

    def __init__(self, channels: int) -> None:
        """Initialize Channel-Attention module."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Multiply inputs with channels attention map."""
        pooled_spatial = self.avg_pool(x)
        channels_attention = self.activation(self.fc(pooled_spatial))
        return x * channels_attention


class SpatialAttention(nn.Module):
    """Spatial-Attention module from http://arxiv.org/pdf/1807.06521."""

    def __init__(self, kernel_size: Literal[3, 7] = 7):
        """Initialize Spatial-Attention module."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv_1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Multiply inputs with spatial attention map."""
        avg_pooled = torch.mean(x, 1, keepdim=True)
        max_pooled = torch.max(x, 1, keepdim=True)[0]
        pooled_feats = torch.cat([avg_pooled, max_pooled], 1)
        spatial_attention = self.activation(self.conv_1(pooled_feats))
        return x * spatial_attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) from http://arxiv.org/pdf/1807.06521."""

    def __init__(self, channels: int, kernel_size: Literal[3, 7] = 7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the forward pass channel attention and spatial attention."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dim: int = 1):
        """Concatenates a list of tensors along a dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.cat(x, self.dim)
