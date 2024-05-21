# RepVGG

2021 | [paper](https://arxiv.org/pdf/2101.03697.pdf) | _RepVGG: Making VGG-style ConvNets Great Again_

Authors present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of $3 × 3$ convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named **RepVGG**.

It is challenging for a plain model to reach a comparable level of performance as the multi-branch architectures. An explanation is that a multi-branch topology, e.g., ResNet, makes the model an implicit ensemble of numerous shallower models, so that training a multi-branch model avoids the gradient vanishing problem. Since the benefits of multi-branch architecture are all for training and the drawbacks are undesired for inference, authors proposed to decouple the training-time multi-branch and inference-time plain architecture via structural re-parameterization, which means converting the architecture from one to another via transforming its parameters. To be specific, a network structure is coupled with a set of parameters, e.g., a conv layer is represented by a 4th-order kernel tensor. If the parameters of a certain structure can be converted into another set of parameters coupled by another structure, we can equivalently replace the former with the latter, so that the overall network architecture is changed.

Specifically, authors construct the training-time RepVGG using _identity_ and $1 × 1$ branches, which is inspired by ResNet but in a different way that the branches can be removed by structural re-parameterization (figures below). After training, they perform the transformation with simple algebra, as an identity branch can be regarded as a degraded $1 × 1$ _conv_, and the latter can be further regarded as a degraded $3 × 3$ _conv_, so that we can construct a single $3 × 3$ kernel with the trained parameters of the original $3 × 3$ kernel, _identity_ and $1 × 1$ branches and batch normalization (BN) layers. Consequently, the transformed model has a stack of $3 × 3$ _conv_ layers, which is saved for test and deployment. Notably, the body of an inference-time RepVGG only has one single type of operator: $3 × 3$ _conv_ followed by ReLU, which makes RepVGG fast on generic computing devices like GPUs. Even better, RepVGG allows for specialized hardware to achieve even higher speed because given the chip size and power consumption, the fewer types of operators we require, the more computing units we can integrate onto the chip. Consequently, an inference chip specialized for RepVGG can have an enormous number of $3 × 3$-ReLU units and fewer memory units (because the plain topology is memory-economical.

There are at least three reasons for using simple ConvNets, they are **fast**, **memory-economical** and **Flexible*

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/15e5d734-fb54-4f18-a076-2638bfada436" alt="rep_vgg_overview" height="450"/>
</p>

## Re-param for Plain Inference-time Model - how to convert a trained block into a single $3 x 3$ _conv_ layer for inference?

Note that authors use BN in each branch before the addition. Formally, we use $W^{(3)}$ (shape: $C_2 × C_1 × 3 × 3$) to denote the kernel of a $3 × 3$ _conv_ layer with $C_1$ input channels and $C_2$ output channels, and $W^{(1)}$ (shape: $C_2 × C_1$ for the kernel of $1 × 1$ branch. We use $μ^{(3)}, σ^{(3)}, γ^{(3)}, β^{(3)}$ as the accumulated _mean_, _standard deviation_ and learned _scaling factor_ and _bias_ of the BN layer following $3 × 3$ _conv_ $3 × 3$ _conv_, $μ^{(1)}, σ^{(1)}, γ^{(1)}, β^{(1)}$ for the BN following $1 × 1$ _conv_, and $μ^{(0)}, σ^{(0)}, γ^{(0)}, β^{(0)}$ for the identity branch. Let $M^{(1)}$ (shape: $N × C_1 × H_1 × W_1$), $M^{(2)}$ (shape: $N × C_2 × H_2 × W_2$ be the input and output, respectively, and $∗$ be the convolution operator. If $C_1 = C_2$, $H_1 = H_2$, $W_1 = W_2$, we have:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/25374565-6b06-41a0-952d-1e6b9d47bc1f" alt="RepVGG_M2_eq" height="120"/>
</p>

Otherwise, we simply use no identity branch, hence the above equation only has the first two terms. Here $b_n$ is the inference-time BN function, formally, $∀1 ≤ i ≤ C_2$,

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/13b7c59e-8a49-4e21-9241-ab5e25223b41" alt="RepVGG_bn_eq" height="60"/>
</p>

We first convert every BN and its preceding _conv_ layer into a _conv_ with a bias vector. Let ${W′, b′}$ be the kernel and bias converted from ${W, μ, σ, γ, β}$, we have

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/e22af79b-3969-4aea-a23c-ad5f04b26313" alt="RepVGG_Wi_eq" height="60"/>
</p>

Then it is easy to verify that $∀1 ≤ i ≤ C$

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f0e8104d-fed5-47ac-9427-5b06ded5d58b" alt="RepVGG_bn_eq_2" height="60"/>
</p>

This transformation also applies to the identity branch because an identity can be viewed as a $1 × 1$ _conv_ with an identity matrix as the kernel. After such transformations, we will have one $3 × 3$ kernel, two $1 × 1$ kernels, and three bias vectors. Then we obtain the final bias by adding up the three bias vectors, and the final $3 × 3$ kernel by adding the $1 × 1$ kernels onto the central point of $3 × 3$ kernel, which can be easily implemented by first zero-padding the two $1 × 1$ kernels to $3 × 3$ and adding the three kernels up, as shown figure below. Note that the equivalence of such transformations requires the $3 × 3$ and $1 × 1$ layer to have the same stride, and the padding configuration of the latter shall be one pixel less than the former. For example, for a $3 × 3$ layer that pads the input by one pixel, which is the most common case, the $1 × 1$ layer should have $padding = 0$.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d60b5a88-7cd0-4581-b746-377b7ecfe96a" alt="RepVGG_Reparam" height="450"/>
</p>

## Architecture

RepVGG is VGG-style in the sense that it adopts a plain topology and heavily uses $3 × 3$ conv, but it does not use _max-pooling_ like VGG because authors desired the body to have only one type of operator. They arranged the $3 × 3$ layers into 5 stages, and the first layer of a stage down-samples with the $stride = 2$. For image classification, they use global average pooling followed by a fully-connected layer as the head. For other tasks, the task-specific heads can be used on the features produced by any layer.

Authors decided the numbers of layers of each stage following three simple guidelines:

* The first stage operates with large resolution, which is time-consuming, so they used only
one layer for lower latency
* The last stage shall have more channels, so they use only one layer to save the parameters
* They put the most layers into the second last stage (with $14 × 14$ output resolution on ImageNet), following ResNet and its recent variants (e.g., ResNet-101 uses 69 layers in its $14 × 14$-resolution stage).
