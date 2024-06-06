# Gold-YOLO

2023 | [paper](https://arxiv.org/pdf/2309.11331) | _Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism_

In the past years, YOLO-series models have emerged as the leading approaches in the area of real-time object detection. Many studies pushed up the baseline to a higher level by modifying the architecture, augmenting data and designing new losses. However, authors find previous models still suffer from information fusion problem, although Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) have alleviated this. Therefore, this study provides an advanced **_Gather- and-Distribute mechanism (GD)_** mechanism, which is realized with convolution and self-attention operations. This new designed model named as **_Gold-YOLO_**, boosts the multi-scale feature fusion capabilities and achieves an ideal balance between latency and accuracy across all model scales. Additionally, authors implement MAE-style pretraining in the YOLO-series for the first time, allowing YOLO-series models could be to benefit from unsupervised pretraining.

In this paper, authors expand on the foundation of _TopFormer_’s theory, propose a novel **_Gather-and-Distribute mechanism (GD)_** for efficient information exchanging in YOLOs by globally fusing multi-level features and injecting the global information into higher levels. This significantly enhances the information fusion capability of the neck without significantly increasing the latency, improving the model’s performance across varying object sizes. Specifically, GD mechanism comprises two branches:

* a shallow gather-and-distribute branch , which extract and fuse feature information via a convolution-based block
* a deep gather- and-distribute branch, which extract and fuse feature information via an attention-based block

To further facilitate information flow, authors introduce a lightweight adjacent-layer fusion module which combines features from neighboring levels on a local scale. The Gold-YOLO architectures surpasses the existing YOLO series, effectively demonstrating the effectiveness of the proposed approach.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8ae99d79-b146-4c6e-b485-0bc6da58314f" alt="gold_yolo" height="300"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3364fef4-e90a-47c8-a0e1-73c74fb8996e" alt="gold_yolo_neck" height="400"/>
</p>

## Preliminaries

The YOLO series neck structure, as depicted in figure above, employs a traditional FPN structure, which comprises multiple branches for multi-scale feature fusion. However, it only fully fuse features from neighboring levels, for other layers information it can only be obtained indirectly "recursively". In figure above, it shows the information fusion structure of the conventional FPN: where existing level-1, 2, and 3 are arranged from top to bottom. FPN is used for fusion between different levels. There are two distinct scenarios when level-1 get information from the other two levels:

1. If level-1 seeks to utilize information from level-2, it can directly access and fuse this information
2. If level-1 wants to use level-3 information, level-1 should recursively call the information fusion module of the adjacent layer. Specifically, the level-2 and level-3 information must be fused first, then level-1 can indirectly obtain level-3 information by combining level-2 information.

This transfer mode can result in a significant loss of information during calculation. Information interactions between layers can only exchange information that is selected by intermediate layers, and not-selected information is discarded during transmission. This leads to a situation where information at a certain level can only adequately assist neighboring layers and weaken the assistance provided to other global layers. As a result, the overall effectiveness of the information fusion may be limited.

To avoid information loss in the transmission process of traditional FPN structures, authors abandon the original recursive approach and construct a novel **_gather-and-distribute mechanism (GD)_**. By using a unified module to gather and fuse information from all levels and subsequently distribute it to different levels, Gold-YOLO not only avoid the loss of information inherent in the traditional FPN structure but also enhance the neck’s partial information fusion capabilities without significantly increasing latency. Gold-YOLO approach thus allows for more effective leveraging of the features extracted by the backbone, and can be easily integrated into any existing backbone-neck-head structure.

In the implementation, the gather and distribute process correspond to three modules: **_Feature Alignment Module (FAM)_**, **_Information Fusion Module (IFM)_**, and **_Information Injection Module (Inject)_**:

* The gather process involves two steps:
    1. FAM collects and aligns features from various levels
    2. IFM fuses the aligned features to generate global information
* Upon obtaining the fused global information from the gather process, the inject module distribute this information across each level and injects it using simple attention operations, subsequently enhancing the branch’s detection capability

To enhance the model’s ability to detect objects of varying sizes, authors developed two branches: low-stage gather-and-distribute branch (**_Low-GD_**) and high-stage gather-and-distribute branch (**_High-GD_**). These branches extract and fuse large and small size feature maps, respectively. As shown in the first figure, the neck’s input comprises the feature maps $B2, B3, B4, B5$ extracted by the backbone, where $Bi ∈ R^{N × C_{Bi} × R_{Bi}}$ . The batch size is denoted by $N$ , the channels by $C$, and the dimensions by $R = H × W$ . Moreover, the dimensions of $R_{B2}, R_{B3}, R_{B4}$, and $R_{B5}$ are $R, \frac{R}{2}, \frac{R}{4}$, and $\frac{R}{8}$, respectively.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/a704e119-2595-40b6-89b0-44f5a7b1dd0d" alt="gold_yolo_GD" height="300"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/4ea88d62-8728-4707-8169-6e2edd5e4eb1" alt="gold_yolo_injection" height="450"/>
</p>

## Low-stage gather-and-distribute branch

In this branch, the output $B2, B3, B4, B5$ features from the backbone are selected for fusion to obtain high resolution features that retain small target information. The structure is shown in figure above.

### Low-stage feature alignment module

In low-stage feature alignment module (**_Low-FAM_**), authors employ the average pooling (_AvgPool_) operation to down-sample input features and achieve a unified size. By resizing the features to the smallest feature size of the group ($R_{B4} = \frac{R}{4}$), authors obtain $F_{align}$. The Low-FAM technique ensures efficient aggregation of information while minimizing the computational complexity for subsequent processing through the transformer module. The target alignment size is chosen based on two conflicting considerations: (1) To retain more low-level information, larger feature sizes are preferable; however, (2) as the feature size increases, the computational latency of subsequent blocks also increases. To control the latency in the neck part, it is necessary to maintain a smaller feature size. Therefore, authors choose the $R_{B4}$ as the target size of feature alignment to achieve a balance between speed and accuracy.

### Low-stage information fusion module

The low-stage information fusion module (**_Low-IFM_**) design comprises multi-layer reparameterized convolutional blocks (**_RepBlock_**) and a split operation. Specifically, RepBlock takes $F_{align}$, $channel = sum(C_{B2}, C_{B3}, C_{B4}, C_{B5})$ as input and produces $F_{fuse}$, $channel = C_{B4} + C_{B5}$. The middle channel is an adjustable value (e.g., 256) to accommodate varying model sizes. The features generated by the RepBlock are subsequently split in the channel dimension into $F_{inj\_P3}$ and $F_{inj\_P4}$, which are then fused with the different level’s feature. The formula is as follows:

```math
F_{align} = Low\_FAM ([B2, B3, B4, B5])
```

```math
F_{fuse} = RepBlock (F_{align})
```

```math
F_{inj\_P3}, F_{inj\_P4} = Split(F_{fuse})
```

### Information injection module

In order to inject global information more efficiently into the different levels, authors draw inspiration from the segmentation experience and employ attention operations to fuse the information, as illustrated in figure above. Specifically, authors input both local information (which refers to the feature of the current level) and global inject information (generated by IFM), denoted as $F_{local}$ and $F_{inj}$ , respectively. They use two different Convs with $F_{inj}$ for calculation, resulting in $F_{global\_embed}$ and $F_{act}$. While $F_{local\_embed}$ is calculated with $F_{local}$ using $Conv$. The fused feature $F_{out}$ is then computed through attention. Due to the size differences between $F_{local}$ and $F_{global}$, authors employ _average pooling_ or _bilinear interpolation_ to scale $F_{global\_embed}$ and $F_{act}$ according to the size of $F_{inj}$ , ensuring proper alignment. At the end of each attention fusion, they add the RepBlock to further extract and fuse the information.

## High-stage gather-and-distribute branch

The High-GD fuses the features { $P3, P4, P5$ } that are generated by the Low-GD, as shown in figure above.

### High-stage feature alignment module

The high-stage feature alignment module (**_High-FAM_**) consists of _avgpool_, which is utilized to reduce the dimension of input features to a uniform size. Specifically, when the size of the input feature is { $R_{P3}, R_{P4}, R_{P5}$ }, _avgpool_ reduces the feature size to the smallest size within the group of features ($R_{P5} = \frac{R}{8}$). Since the transformer module extracts high-level information, the pooling operation facilitates information aggregation while decreasing the computational requirements for the subsequent step in the Transformer module.

### High-stage information fusion module

The high-stage information fusion module (**_High-IFM_**) comprises the transformer block and a splitting operation, which involves a three-step process

1. the $F_{align}$, derived from the High-FAM, are combined using the transformer block to obtain the $F_{fuse}$
2. The $F_{fuse}$ channel is reduced to $sum(C_{P4}, C_{P5}$) via a $1 × 1$ _conv_ operation
3. The $F_{fuse}$ is partitioned into $F_{inj\_N4}$ and $F_{inj\_N5}$ along the channel dimension through a splitting operation, which is subsequently employed for fusion with the current level feature.

The formula is as follows:


```math
F_{align} = High\_FAM ([P3, P4, P5])
```

```math
F_{fuse} = Transformer(F_{align})
```

```math
F_{inj\_N4}, F_{inj\_N5} = Split(Conv_{1×1}(F_{fuse}))
```

The transformer fusion module in equations above comprises several stacked transformers, with the number of transformer blocks denoted by $L$. Each transformer block includes a _multi-head attention block_, a _Feed-Forward Network (FFN)_, and _residual connections_. To configure the multi-head attention block, authors adopt the same settings as LeViT, assigning head dimensions of keys $K$ and queries $Q$ to $D$ (e.g., 16) channels, and $V = 2D$ (e.g., 32) channels. In order to accelerate inference, authors substitute the velocity-unfriendly operator, _Layer Normalization_, with _Batch Normalization_ for each convolution, and replace all _GELU_ activations with _ReLU_. This minimizes the impact of the transformer module on the model’s speed. To establish the _Feed-Forward Network_, authors follow the methodologies presented in [paper_1](https://arxiv.org/pdf/2103.11816) and [paper_2](https://arxiv.org/pdf/2106.03650) for constructing the FFN block. To enhance the local connections of the transformer block, authors introduce a _depth-wise convolution_ layer between the two $1 x 1$ convolution layers. They also set the expansion factor of the FFN to 2, aiming to balance speed and computational cost.

### Information injection module

The information injection module in High-GD is exactly the same as in Low-GD. In high stage, $F_{local}$ is equal to $Pi$, so the formula is as follows

```math
F_{global\_act\_Ni} = resize(Sigmoid(Conv_{act}(F_{inj\_Ni})))
```

```math
F_{global\_embed\_Ni} = resize(Conv_{global\_embed\_Ni}(F_{inj\_Ni}))
```

```math
F_{att\_fuse\_Ni} = Conv_{local\_embed\_Ni}(Pi) ∗ F_{ing\_act\_Ni} + F_{global\_embed\_Ni}
```

```math
Ni = RepBlock(F_{att\_fuse\_Ni})
```

## Enhanced cross-layer information flow

Authors have achieved better performance than existing methods using only a global information fusion structure. To further enhance the performance, they drew inspiration from the PAFPN module in YOLOv6 and introduced an **_Inject-LAF_** module. This module is an enhancement of the injection module and includes a lightweight adjacent layer fusion (**_LAF_**) module that is added to the input position of the injection module. To achieve a balance between speed and accuracy, authors designed two LAF models:

* LAF low-level model, used for low-level injection (merging features from adjacent two layers)
* LAF high-level model, used for high-level injection (merging features from adjacent one layer)

Their structure is shown in figure above. To ensure that feature maps from different levels are aligned with the target size, the two LAF models in Gold-YOLO implementation utilize only three operators: _bilinear interpolation_ to up-sample features that are too small, _average pooling_ to down-sample features that are too large, and $1 x 1$ _convolution_ to adjust features that differ from the target channel. The combination of the LAF module with the information injection module in Gold-YOLO model effectively balances the between accuracy and speed. By using simplified operations, authors are able to increase the number of information flow paths between different levels, resulting in improved performance without significantly increased the latency.

## Masked image modeling pre-training

Recent methods such as BEiT, MAE and SimMIM, have demonstrated the effectiveness of masked image modeling (MIM) for vision tasks. However, these methods are not specifically tailored for convolutional networks (_convnets_). SparK and ConvNeXt-V2 are pioneers in exploring the potential of masked image modeling for convnets. In this study, authors adopt MIM Pre-training following the SparK’s methodology, which successfully identifies and overcomes two key obstacles in extending the success of MAE-style pretraining to convolutional networks (convnets). These challenges include the convolutional operations’ inability to handle irregular and randomly masked input images, as well as the inconsistency between the single-scale nature of BERT pretraining and the hierarchical structure of convnets. To address the first issue, unmasked pixels are treated as sparse voxels of 3D point clouds and employ sparse convolution for encoding. For the latter issue, a hierarchical decoder is developed to reconstruct images from multi-scale encoded features. The framework adopts a UNet-style architecture to decode multi-scale sparse feature maps, where all spatial positions are filled with embedded masks. Authors pretrain our model’s backbone on ImageNet 1K for multiple Gold-YOLO models, and results in notable improvements.
