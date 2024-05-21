# EfficientDet

2020 | [paper](https://arxiv.org/pdf/1911.09070) | _EfficientDet: Scalable and Efficient Object Detection_

In this paper, authors systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, they propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, they propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints.

Based on the one-stage detector paradigm, authors examine the design choices for backbone, feature fusion, and class/box network, and identify two main challenges:

* _Efficient multi-scale feature fusion_ – FPN has been widely used for multiscale feature fusion. Recently, PANet, NAS-FPN, and other studies have developed more network structures for cross-scale feature fusion. While fusing different input features, most previous works simply sum them up without distinction; however, since these different input features are at different resolutions, it can be observed they usually contribute to the fused output feature unequally. To address this issue, authors propose a simple yet highly effective **Weighted Bi-Directional Feature Pyramid Network** (**BiFPN**), which introduces learnable weights to learn the importance of different input features, while repeatedly applying top-down and bottom-up multi-scale feature fusion.
* _model scaling_ - while previous works mainly rely on bigger backbone networks or larger input image sizes for higher accuracy, authors observed that scaling up feature network and box/class prediction network is also critical when taking into account both accuracy and efficiency. Inspired by recent works (EfficientNet, same authors), they propose a compound scaling method for object detectors, which jointly scales up the resolution/depth/width for all backbone, feature network, box/class prediction network.

Authors also combinined EfficientNet backbones with the proposed BiFPN and compound scaling, and have developed a new family of object detectors, named **EfficientDet**, which consistently achieve better accuracy with much fewer parameters and FLOPs than previous object detectors.

## BiFPN - Bidirectional cross-scale connections and weighted feature fusion

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/0f322730-fc06-430d-bb8e-138c60345e15" alt="efficient_det_bifpn_vs_all" height="400"/>
</p>

### Problem formulation

Multi-scale feature fusion aims to aggregate features at different resolutions. Formally, given a list of multi-scale features $P^{in} = (P^{in}_{l_1} , P^{in}_{l_2} , ...), where $P^{in}_{l_i}$ represents the feature at level $l_i$, the goal is to find a transformation $f$ that can effectively aggregate different features and output a list of new features: $P^{out} = f(P^{in})$. As a concrete example, figure above shows the conventional top-down FPN. It takes level 3-7 input features $P^{in} = (P^{in}_3 , ... P^{in}_7)$, where $P^{in}_i$ represents a feature level with resolution of $1/{2^{i}}$ of the input images. For instance, if input resolution is $640 x 640$, then $P^{in}_3$ represents feature level 3 $(640/{2^3} = 80)$ with resolution $80 x 80$, while $P^{in}_7$ represents feature level 7 with resolution $5 x 5$. The conventional FPN aggregates multi-scale features in a top-down manner:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/33205b9f-5f82-4105-a61a-2cbdb356de59" alt="efficient_det_fpn_eq" height="150"/>
</p>

where _Resize_ is usually a upsampling or downsampling operation for resolution matching, and _Conv_ is usually a convolutional operation for feature processing.

### Cross-Scale Connections

Conventional top-down FPN is inherently limited by the one-way information flow. To address this issue, PANet adds an extra bottom-up path aggregation network, as shown in Figure above (b). Recently, NAS-FPN employs neural architecture search to search for better cross-scale feature network topology, but it requires thousands of GPU hours during search and the found network is irregular and difficult to interpret or modify, as shown in figure above (c). By studying the performance and efficiency of these three networks, one can observe that PANet achieves better accuracy than FPN and NAS-FPN, but with the cost of more parameters and computations. To improve model efficiency, this paper proposes several optimizations for cross-scale connections: 

* First - remove those nodes that only have one input edge. The intuition is simple: if a node has only one input edge with no feature fusion, then it will have less contribution to feature network that aims at fusing different features. This leads to a simplified bi-directional network
* Second - add an extra edge from the original input to output node if they are at the same level, in order to fuse more features without adding much cost
* Third - unlike PANet that only has one top-down and one bottom-up path, treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion. 

With these optimizations, the new feature network is named bidirectional feature pyramid network (BiFPN), as shown in figures above and below.

### Weighted Feature Fusion

When fusing features with different resolutions, a common way is to first resize them to the same resolution and then sum them up. PAN introduces global self-attention upsampling to recover pixel localization. All previous methods treat all input features equally without distinction. However, authors observed that since different input features are at different resolutions, they usually contribute to the output feature unequally. To address this issue, they proposed to add an additional weight for each input, and let the network to learn the importance of each input feature. Based on this idea, they considered three weighted fusion approaches:

* **Unbounded fusion** - $O = \sum_{i} w_i * I_i$, where $w_i$ is a learnable weight.
* **Softmax-based fusion** - $O = \sum_{i} \frac{e^{w_i}}{\sum_{j}e^{w_j}} * I_i$ - an intuitive idea is to apply softmax to each weight, such that all weights are normalized to be a probability with value range from 0 to 1, representing the importance of each input. However, as shown later, the extra softmax leads to significant slowdown on GPU hardware
* **Fast normalized fusion** - $O = \sum_{i} \frac{w_i}{\epsilon + \sum_{j}w_j} * I_i$, where
$w_i ≥ 0$ is ensured by applying a ReLU after each $w_i$, and $\epsilon = 0.0001$ is a small value to avoid numerical instability. Similarly, the value of each normalized weight also falls between 0 and 1, but since there is no softmax operation here, it is much more efficient. The ablation study shows this fast fusion approach has very similar learning behavior and accuracy as the softmax-based fusion, but runs up to 30% faster on GPUs.

The final BiFPN integrates both the bidirectional cross-scale connections and the fast normalized fusion. As a concrete example, here is described the two fused features at
level 6 for BiFPN shown in figure above (d):

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3b1d6b83-16aa-4852-a5aa-e76f21dd02a6" alt="efficient_det_bifpn_eq" height="120"/>
</p>

where $P^{td}_6$ is the intermediate feature at level 6 on the top-down pathway, and $P^{out}_6$ is the output feature at level 6 on the bottom-up pathway. All other features are constructed in a similar manner. Notably, to further improve the efficiency, authors used depthwise separable convolution for feature fusion, and add batch normalization and activation after each convolution.

## EfficientDet Architecture

Based on the BiFPN, authors have developed a new family of detection models named EfficientDet. Figure below shows the overall architecture of EfficientDet, which largely follows the one-stage detectors paradigm. Authors employ ImageNet-pretrained EfficientNets as the backbone network. The proposed BiFPN serves as the feature network, which takes level 3-7 features ${P_3, P_4, P_5, P_6, P_7}$ from the backbone network and repeatedly applies top-down and bottom-up bidirectional feature fusion. These fused features are fed to a class and box network to produce object class and bounding box predictions respectively. The class and box network weights are shared across all levels of features.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/ea80cb8a-b8e5-47fe-a391-55bf993dd64f" alt="efficient_det" height="400"/>
</p>

## Compound Scaling

Compound scaling is a method proposed in [EfficientNet](https://arxiv.org/pdf/1905.11946v5),  that uniformly scales all dimensions of network width, depth, and resolution using a fixed ratio. It involves adjusting the network depth, width, and image resolution simultaneously with constant coefficients to achieve a balance between these dimensions. The compound scaling method simplifies the process of scaling ConvNets by determining how many more resources are available for model scaling and how to allocate these resources to width, depth, and resolution. By using compound scaling, the total FLOPS of the network can be approximately increased by a factor determined by the scaling coefficients

> **_NOTE:_** Most of the new YOLO approaches uses the BiFPN module and/or the feature pyramid weighting mechanism.
