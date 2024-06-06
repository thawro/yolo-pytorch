# YOLO-MS

2023 | [paper](https://arxiv.org/pdf/2308.05480) | _YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection_

Authors aim at providing the object detection community with an efficient and performant object detector, termed **_YOLO-MS_**. The core design is based on a series of investigations on how convolutions with different kernel sizes affect the detection performance of objects at different scales. The outcome is a new strategy that can strongly enhance multi-scale feature representations of real-time object detectors. To verify the effectiveness of this strategy, authors build a network architecture, termed **_YOLO-MS_**. They train the YOLO-MS on the MS COCO dataset from scratch without relying on any other large-scale datasets, like ImageNet, or pre-trained weights. Without bells and whistles, YOLO-MS outperforms the recent SoTA real-time object detectors, including YOLO-v7 and RTMDet, when using a comparable number of parameters and FLOPs

Real-time object detection, exemplified by the YOLO series, has found significant applications in industries, particularly for edge devices, such as drones and robotics. Different from previous heavy object detectors, real-time object detectors aim to pursue an optimal trade-off between speed and accuracy. In pursuit of this objective, a huge amount of work has been proposed: from the first generation of _DarkNet_ to _CSPNet_, and then to the recent extended _ELAN_, the architectures of real-time object detectors have experienced great changes accompanied by the fast growth of performance. Despite the impressive performance, recognizing objects at different scales remains a fundamental challenge for real-time object detectors. This motivates authors to design a robust encoder architecture for learning expressive multi-scale feature representations. Specifically, they consider encoding multi-scale features for real-time object detection from two new perspectives:

* From a local view, they design an **_MS-Block_** with a simple yet effective hierarchical feature fusion strategy. Inspired by _Res2Net_, authors introduce multiple branches in MS-Block to perform feature extraction, but differently, they use an inverted bottleneck block with _depth-wise convolution_ to enable the efficient use of large kernels
* From a global view, they propose gradually increasing the kernel size of convolutions as the network goes deeper:
  * use small kernel convolutions in shallow layers to process the high-resolution features more efficiently
  * use large kernel convolutions in deep layers to capture wide-range information

Based on the above design principles, authors present the real-time object detector, termed **_YOLO-MS_**.

## Multi-Scale Building Block Design

yolo_ms_blocks

<p align="center">
  <img src="" alt="" height="450"/>
</p>

CSP Block is a stage-level gradient path-based network that balances the gradient combinations and computational cost. It is a fundamental building block widely utilized in the YOLO series There are several variants having been proposed, including:

* Original CSP Block in YOLOv4 and YOLOv5 (figure above, (a))
* CSPVoVNet in Scaled-YOLOv4
* ELAN in YOLOv7 (figure above, (b))
* large kernel unit proposed in RTMDet

A key aspect that has been overlooked by the aforementioned real-time detectors is how to encode multi-scale features in the basic building blocks. One of the powerful design principles is Res2Net, which aggregates features from different hierarchies to enhance the multi-scale representations. However, this principle does not thoroughly explore the role of large kernel convolutions, which have demonstrated effective in CNN-based models for visual recognition tasks. The main obstacle in incorporating large kernel convolutions into Res2Net is the computational overhead they introduce as the building block adopts the standard convolutions. In YOLO-MS method, authors propose to replace the standard $3 × 3$ _convolution_ with the inverted bottleneck layer to enjoy large-kernel convolutions.

### MS-Block

Based on the earlier analysis, authors propose a novel block with a hierarchical feature fusion strategy, named **_MS-Block_**, to enhance the capability of real-time object detectors in extracting multi-scale features while maintaining a fast inference speed. The specifics of MS-Block are illustrated in figure above (c). Suppose $X ∈ R^{H × W × C}$ refers to the input feature. After the transition via a $1 × 1$ _convolution_, the channel dimension of $X$ is increased to $n × C$. Then, $X$ is split into $n$ distinct groups, denoted as {$X_i$} where $i ∈ 1, 2, 3, ..., n$. To lower the computational cost, authors choose $n$ to be 3. Note that apart from $X_1$, each other group goes through an inverted bottleneck layer, denoted by $IB_{k×k}(·)$ where $k$ refers to kernel size, to attain $Y_i$. The mathematical representation of $Y_i$ can be described as follows:

$$ Y_i = \begin{cases} X_i & \text{i = 1}\\ IB_{k×k}(Y_{i−1} + X_i) & \text{i > 1}\\ \end{cases} $$

Following the formula, authors do not connect the inverted bottleneck layer to $X_1$, allowing it to act as a cross-stage connection and retain information from the previous layers. Finally, they concatenate all splits and apply a $1 × 1$ _convolution_ to conduct an interaction among all splits, each of which encodes features at different scales. This $1 × 1$ _convolution_ is also used to adjust the channel numbers when the network goes deeper.

## Heterogeneous Kernel Selection Protocol

yolo_ms_HKS

<p align="center">
  <img src="" alt="" height="550"/>
</p>

In addition to the design of building block, authors also delve into the usage of convolution from a macro perspective. Previous real-time object detectors adopt homogeneous _convolutions_ (i.e., _convolutions_ with the same kernel size) in different encoder stages but authors argue that this is not the optimal option for extracting multi-scale semantic information.

In a pyramid architecture, high-resolution features extracted from the shallow stages of the detectors are commonly used to capture fine-grained semantics, which will be used to detect small objects. On the contrary, low-resolution features from the deeper stages of the network are utilized to capture high-level semantics, which will be used to detect large objects. If we adopt uniform small-kernel _convolutions_ for all stages, the _Effective Receptive Field (ERF)_ in the deep stage is limited, affecting the performance on large objects. Incorporating large-kernel _convolutions_ in each stage can help address this limita- tion. However, large kernels with a large ERF can encode broader regions, which increases the probability of including contaminative information outside the small objects and decreases the inference speed.

In this work, authors propose to leverage heterogeneous _convolutions_ in different stages to assist in capturing richer multi-scale features. To be specific, they leverage the smallest-kernel convolution in the first stage of the encoder while the largest-kernel convolution is in the last stage. Subsequently, they incrementally increase the kernel size for the middle stages, aligning it with the increment of feature resolution. This strategy allows for extracting both fine-grained and coarse-grained semantic information, enhancing the multi-scale feature representation capability of the encoder.

As presented in figure above, authors assign values of $k$ to $3, 5, 7$, and $9$ from the shallow stage to the deep stage in the encoder. They call this **_Heterogeneous Kernel Selection (HKS) Protocol_**. HKS protocol is able to enlarge the receptive fields in the deep stages without introducing any other impact to the shallow stages. Additionally, HKS not only helps encode richer multi-scale features but also ensures efficient inference.

As shown in table below, applying large-kernel convolutions to high-resolution features is computationally expensive. However, the proposed HKS protocol adopts large-kernel convolution on low-resolution features, thereby substantially reducing computational costs compared to only using large-kernel convolutions. In practice, authors empirically found that YOLO-MS with HKS protocol achieves nearly the same inference speed as that only using _depth-wise_ $3 × 3$ _convolutions_.

yolo_ms_HKS_table

<p align="center">
  <img src="" alt="" height="250"/>
</p>

## Architecture

As shown in figure above, the backbone of YOLO-MS consists of four stages, each of which is followed by a $3 × 3$ _convolution_ with stride 2 for downsampling. Authors add an _SPP_ block after the third stage as done in RTMDet. Authors use PAFPN as neck to build feature pyramid upon the encoder. It fuses multi-scale features extracted from different stages in the backbone. The basic building blocks used in the neck are also the proposed MS-Block, in which $3 × 3$ _depth-wise convolutions_ are used for fast inference. In addition, to achieve a better trade-off between speed and accuracy, YOLO-ms halves the channel depth of the multi-level features from the backbone. Thee variants of YOLO-MS are presented, namely YOLO-MS-XS, YOLO-MS-S, and YOLO-MS, which differs in channels number used for each stage.
