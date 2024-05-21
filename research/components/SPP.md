# SPP

2015 | [paper](https://arxiv.org/pdf/1406.4729.pdf) | _Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition_

Authors of the paper introduced a Spatial Pyramid Pooling (**SPP**) layer to remove the fixed-size constraint of the network. Specifically, added an SPP layer on top of the last convolutional layer. The SPP layer pools the features and generates fixed length outputs, which are then fed into the fullyconnected layers (or other classifiers). In other words, we perform some information “aggregation” at a deeper stage of the network hierarchy (between convolutional layers and fully-connected layers) to avoid the need for cropping or warping at the beginning

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/20e4130d-8b7d-40be-9e03-3de729989654" alt="SPP" height="350"/>
</p>

Spatial pyramid pooling (also knows as spatial pyramid matching or SPM), as an extension of the Bag-of-Words (BoW) model, is one of the most successful methods in computer vision. It partitions the image into divisions from finer to coarser levels, and aggregates local features in them. SPP has long been a key component in the leading and competition-winning systems for classification and detection  before the prevalence of CNNs. Nevertheless, SPP has not been considered in the context of CNNs before this paper. Authors noted that SPP has several remarkable properties for deep CNNs:

* SPP is able to generate a fixed-length output regardless of the input size, while the sliding window pooling used in the previous deep networks cannot
* SPP uses multi-level spatial bins, while the sliding window pooling uses only a single window size. Multi-level pooling has been shown to be robust to object deformations
* SPP can pool features extracted at variable scales thanks to the flexibility of input scales. Through experiments authors have shown that all these factors elevate the recognition accuracy of deep networks.

SPP-net not only makes it possible to generate representations from arbitrarily sized images/windows for testing, but also allows us to feed images with varying sizes or scales during training. Training with variable-size images increases scale-invariance and reduces over-fitting. 

> **_NOTE:_** Most of the YOLO approaches uses only the idea of modified **SPP block** (e.g. SPPF in YOLOv5) - applying _max-pooling_ operations with different kernel sizes (with padding) and then concatenating the results to form a single feature map, which is passed to next blocks.