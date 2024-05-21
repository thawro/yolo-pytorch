# Scaled YOLOv4

2021 | [paper](https://arxiv.org/pdf/2011.08036.pdf) | _Scaled-YOLOv4: Scaling Cross Stage Partial Network_
Scaled version of YOLOv4 built on top of the [YOLOv4](#yolo-v4). Authors have shown that the YOLOv4 object detection neural network based on the [CSP](#csp) approach, scales both up and down and is applicable to small and large networks while maintaining optimal speed and accuracy. They proposed a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network.

## How it works

When designing the efficient model scaling methods, authors main principles are:

* when the scale is up, the lower the quantitative cost we want to increase, the better
* when the scale is down, the higher the quantitative cost we want to decrease, the better

The [CSP](https://arxiv.org/pdf/1911.11929) can be applied to various CNN architectures, while reducing the amount of parameters and computations. In addition, it also improves accuracy and reduces inference time. CSP connection is extremely efficient, simple, and can be applied to any neural network. The idea is:
* half of the output signal goes along the main path (generates more semantic information with a large receiving field)
* and the other half of the signal goes bypass (preserves more spatial information with a small perceiving field)

Due to that, authors used CSP-ized models to perform model scaling. 

In case of the _tiny_ model, authors wanted the computations to be even smaller, so they adapted the [OSANet](https://arxiv.org/pdf/1904.09730) upgraded by [CSP](https://arxiv.org/pdf/1911.11929) module to create the CSPOSANet backbone

Usually, it is possible to adjust the scaling factors of an object detector’s input, backbone, and neck. The potential scaling factors that can be adjusted are summarized:

* Input: _size_
* Backbone: _width_, _depth_, _#stage_ (number of stages)
* Neck: _width_, _depth_, _#stage_ (number of stages)

The biggest difference between image classification and object detection is that the former only needs to identify the category of the largest component in an image, while the latter needs to predict the position and size of each object in an image. In one-stage object detector, the feature vector corresponding to each location is used to predict the category and size of an object at that location. The ability to better predict the size of an object basically depends on the receptive field of the feature vector. In the CNN architecture, the thing that is most directly related to receptive field is the stage, and the feature pyramid network (FPN) architecture tells us that higher stages are more suitable for predicting large objects. When the input image size is increased, if one wants to have a better prediction effect for large objects, he/she must increase the depth or number of stages of the network. When performing scaling up, authors first perform compound scaling on input size, and number of stages, and then according to real-time requirements, we further perform scaling on depth and width respectively

## Model architecture

* **Backbone** - in the design of CSPDarknet53, the computation of down-sampling convolution for cross-stage process is not included in a residual block. Therefore, we can deduce that CSPDarknet stage will have a better computational advantage over Darknet stage only when number of layers in a block is greater than 1. The number of residual layer owned by each stage in CSPDarknet53 is 1-2-8-8-4 respectively. In order to get a better speed/accuracy trade-off, the first CSP stage is converted into original Darknet residual layer
* **Neck** - in order to effectively reduce the amount of computation, [CSP](https://arxiv.org/pdf/1911.11929) is applied to the [PAN](https://arxiv.org/pdf/1803.01534) architecture in YOLOv4. The computation list of a PAN architecture is illustrated below (left). It mainly integrates the features coming from different feature pyramids, and then passes through two sets of reversed Darknet residual layer without shortcut connections. After applying [CSP](https://arxiv.org/pdf/1911.11929) (right), the computation is cut down by 40% and the neck is called CSPPAN.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/dfa8d3b9-34d6-4244-a370-a3f2834f9037" alt="reversed_CSP_dark_layers_SPP" height="300"/>
</p>

* **SPP** - the [SPP](https://arxiv.org/pdf/1406.4729) module was originally inserted in the middle position of the first computation list group of the neck. Therefore, in Scaled YOLOv4, the SPP is also inserted in the middle position of the first computation list group of the CSPPAN

Improvements in Scaled YOLOv4 over YOLOv4:
* Improved network architecture - backbone is optimized and Neck (PAN) uses CSP connections and Mish activation
* Exponential Moving Average (EMA) is used during training — this is a special case of Stochastic Weight Averaging ([SWA](https://arxiv.org/pdf/1803.05407), [SWA-PyTorch](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/))
* Multiresolution - For each resolution of the network, a separate neural network is trained (in YOLOv4, only one neural network was trained for all resolutions)
* Improved objectness normalizers in prediction layers
* Changed activations for Width and Height, which allows faster network training
* Pre Processing - the Letter Box pre processing is used (?) to keep the aspect ratio of the input image (tiny version doesnt use Letter Box)

### YOLOv4-tiny

YOLOv4-tiny is designed for low-end GPU device. The _CSPOSANet_ with _PCB_ (Partial in Computational Block) architecture is used as backbone. Growth rate (_g_) is set to _b/2_  (_b_ - base width) and it grows to _b/2 + kg = 2b_ at the end (_k_ - number of layers). Through calculation,  _k = 3_ is set, and its architecture and the computational block of YOLOv4-tiny is shown below. As for the number of channels of each stage and the part of neck, the design of YOLOv3-tiny is followed.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/81da6cf6-6ce8-4e41-a5d4-aa47038c4b17" alt="yolo_v4_tiny" height="300"/>
</p>

### YOLOv4-large

YOLOv4-large is designed for cloud GPU, the main purpose is to achieve high accuracy for object detection. Authors designed a fully CSP-ized model YOLOv4-P5 and scaled it up to YOLOv4-P6 and YOLOv4-P7. The structure of YOLOv4-P5, YOLOv4-P6, and YOLOv4-P7 is shown below. The compound scaling of input size and number of stages was performed. The depth scale of each stage was set to to 2^dsi , and ds to [1, 3, 15, 15, 7, 7, 7]. Finally, the width scaling was performed with the use of inference time as constraint.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/9c5e0666-d82d-4944-8877-889783e02090" alt="yolo_v4_large" height="400"/>
</p>

## Training details

The models are trained directly on MS COCO dataset (no pre-training on ImageNet). The training choices include:

* SGD optimizer
* Training time:
  * YOLOv4-tiny: 600 epochs,
  * YOLOv4-CSP: 300 epochs, 
  * YOLOv4-large: 300 epochs first and then followed by using stronger data augmentation method to train 150 epochs
* Hyperparameters (anchors, learning rate, degree of data augmentation) were found using k-means and genetic algorithms

## Loss functions

There are different Losses in YOLOv3, YOLOv4 and Scaled-YOLOv4:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/7005d806-0928-4686-8dfa-cb79252a5ee4" alt="scaled_yolo_v4_losses" height="250"/>
</p>

* for _bx_ and _by_ — this eliminates grid sensitivity in the same way as in YOLOv4, but more aggressively
* for _bw_ and _bh_ — this limits the size of the bounded-box to _4*Anchor_size_
