# Deformable Convolution

2017 | [paper](https://arxiv.org/pdf/1703.06211v3) | _Deformable Convolutional Networks_

Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in their building modules. In this work, authors introduce two new modules to enhance the transformation modeling capability of CNNs, namely, _deformable convolution_ and _deformable RoI pooling_. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from the target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks

* **Deformable Convolution** - it adds 2D offsets to the regular grid sampling locations in the stan-
dard convolution. It enables free form deformation of the sampling grid. It is illustrated in figure above. The offsets are learned from the preceding feature maps, via additional convolutional layers. Thus, the deformation is conditioned on the input features in a local, dense, and adaptive manner

* **Deformable RoI pooling**  - it adds an offset to each bin position in the regular bin partition of the previous RoI pooling. Similarly, the offsets are learned from the preceding feature maps and the RoIs, enabling adaptive part localization for objects with different shapes. Both modules are light weight. They add small amount of parameters and computation for the offset learning. They can readily replace their plain counterparts in deep CNNs and can be easily trained end-to-end with standard back- propagation. The resulting CNNs are called deformable convolutional networks, or deformable ConvNets

## Operation

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/70c791c6-fc3a-497f-a5a0-3054c6382703" alt="DeformConv_offsets" height="350"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/cc97f072-7dde-4493-a65e-e828f13a3178" alt="DeformConv_block" height="350"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/4ac49c06-2352-4fb3-956e-86f7f9a642ee" alt="DeformConv_receptive_field" height="350"/>
</p>

The 2D convolution consists of two steps: 

1. sampling using a regular grid R over the input feature map x
2. summation of sampled values weighted by $w$

The grid $R$ defines the receptive field size and dilation. For example

$$ R = {(−1, −1), (−1, 0), . . . , (0, 1), (1, 1)}$$

defines a $3 × 3$ kernel with $dilation = 1$. For each location $p_0$ on the output feature map $y$, we
have

$$ y(p_0) = \sum_{p_n ∈ R}w(p_n) · x(p_0 + p_n) $$

where $p_n$ enumerates the locations in $R$. In deformable convolution, the regular grid $R$ is augmented with offsets ${∆p_n | n = 1, ..., N }$, where $N = |R|$. Equation above becomes

$$ y(p_0) = \sum_{p_n ∈ R}w(p_n) · x(p_0 + p_n + ∆p_n) $$

,the sampling is on the irregular and offset locations $p_n + ∆p_n$. As the offset $∆p_n$ is typically fractional, Equation above is implemented via bilinear interpolation as:

$$ x(p) = \sum_{q} G(q, p) · x(q) $$

where $p$ denotes an arbitrary (fractional) location ($p = p_0 + p_n + ∆p_n$ for Equation above), $q$ enumerates all integral spatial locations in the feature map $x$, and $G(·, ·)$ is the bilinear interpolation kernel. Note that $G$ is two dimensional. It is separated into two one dimensional kernels as

$$ G(q, p) = g(q_x, p_x) · g(q_y , p_y) $$

where $g(a, b) = max(0, 1 − |a − b|)$. Eq. (3) is fast to compute as $G(q, p)$ is non-zero only for a few $q_s$.

As illustrated in figures above, the offsets are obtained by applying a convolutional layer over the same input feature map. The convolution kernel is of the same spatial resolution and dilation as those of the current convolutional layer (e.g., also $3 × 3$ with dilation 1 in figures). The output offset fields have the same spatial resolution with the input feature map. The channel dimension $2N$ corresponds to $N$ 2D offsets. During training, both the convolutional kernels for generating the output features and the offsets are learned simultaneously. To learn the offsets, the gradients are backpropagated through the bilinear operations.

## Deformable ConvNets

To integrate deformable ConvNets with the SoTA CNN architectures, authors note that these architectures consist of two stages. First, a deep fully convolutional network generates feature maps over the whole input image. Second, a shallow task specific network generates results from the feature maps.