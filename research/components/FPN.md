# FPN

2017 | [paper](https://arxiv.org/pdf/1612.03144.pdf) | _Feature Pyramid Networks for Object Detection_

Authors of the paper exploited the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (**FPN**), shows significant improvement as a generic feature extractor in several applications. 

Pyramid approaches comparison:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/4ee4eeee-c9a4-4873-8dce-150ab73c7262" alt="pyramids_comparison" height="350"/>
</p>

A deep ConvNet computes a feature hierarchy layer by layer, and with subsampling layers the feature hierarchy has an inherent multi- scale, pyramidal shape. This in-network feature hierarchy produces feature maps of different spatial resolutions, but introduces large semantic gaps caused by different depths. The high-resolution maps have low-level features that harm their representational capacity for object recognition

FPN is a feature pyramid that has strong semantics at all scales and is built quickly from a single input image scale. It combines low-resolution, semantically strong features with high-resolution, semantically weak features via a top-down pathway and lateral connections.

FPN takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps at multiple levels, in a fully convolutional fashion. This process is independent of the backbone convolutional architectures. The construction of the pyramid involves a bottom-up pathway, a top-down pathway, and lateral connections, as introduced in the following:

FPN compared to [similar](https://arxiv.org/pdf/1603.08695) approach:
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8b5553e8-d1f0-4a4e-afab-91cd51e6b079" alt="FPN" height="350"/>
</p>

* **Bottom-up pathway** - the feedforward computation of the backbone ConvNet, which computes a feature hierarchy consisting of feature maps at several scales with a scaling step of $2$. There are often many layers producing output maps of the same size - these layers are in the same network stage. For the feature pyramid, one pyramid level is defined for each stage. The output of the last layer of each stage is chosen as the reference set of feature maps, which will be enriched to create the pyramid. This choice is natural since the deepest layer of each stage should have the strongest features. Specifically, for ResNets the feature activations output by each stage’s last residual block are used. The output of these last residual blocks are denoted as ${C_2, C_3, C_4, C_5}$ for _conv2_, _conv3_, _conv4_, and _conv5_ outputs, with corresponding strides of ${4, 8, 16, 32}$ pixels with respect to the input image. The _conv1_ is not included into the pyramid due to its large memory footprint.

* **Top-down pathway and lateral connections** - it hallucinates higher resolution features by upsampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels. These features are then enhanced with features from the bottom-up pathway via lateral connections. Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times. The building block shown below constructs the top-down feature maps.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/c20d0886-7ea0-4e47-89ef-3655d6e8e5df" alt="FPN_block" height="350"/>
</p>

With a coarser-resolution feature map, the spatial resolution is upsampled by a factor of $2$ (using nearest neighbor upsampling for simplicity). The upsampled map is then merged with the corresponding bottom-up map (which undergoes a $1 × 1$ convolutional layer to reduce channel dimensions) by element-wise addition. This process is iterated until the finest resolution map is generated. Iteration is started by simply attaching a $1 × 1$ convolutional layer on $C_5$ to produce the coarsest resolution map. Finally, a $3 × 3$ convolution is appended on each merged map to generate the final feature map, which is to reduce the aliasing effect of upsampling. This final set of feature maps is called ${P_2, P_3, P_4, P_5}$, corresponding to ${C_2, C_3, C_4, C_5}$ that are respectively of the same spatial sizes. Because all levels of the pyramid use shared classifiers/regressors as in a traditional featurized image pyramid, the feature dimension (numbers of channels, denoted as $d$) is fixed in all the feature maps. $d = 256$ in this paper and thus all extra convolutional layers have 256-channel outputs. There are no non-linearities in these extra layers, which has been empirically found to have minor impacts. The more sophisticated blocks (e.g., using multi-layer residual blocks as the connections) have been tested and observed marginally better results.

> **_NOTE:_** Most of the YOLO approaches uses the idea of connecting the backbone path (**Bottom-up pathway**) with the neck path **Top-down pathway** by the use of the **lateral connections**. The FPN is further improved to PAN and finally to BiFPN neck configurations.
