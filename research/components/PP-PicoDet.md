# PP-PicoDet

2021 | [paper](https://arxiv.org/pdf/2111.00902) | _PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices_

The better accuracy and efficiency trade-off has been a challenging problem in object detection. In this work, authors are dedicated to studying key optimizations and neural network architecture choices for object detection to improve accuracy and efficiency. They investigate the applicability of the anchor-free strategy on lightweight object detection models. Authors enhance the backbone structure and design the lightweight structure of the neck, which improves the feature extraction ability of the network. PP-PicoDet also improves label assignment strategy and loss function to make training more stable and efficient. PP-PicoDet achieves superior performance on object detection for mobile devices.

The problem is that lightweight anchor-free detectors usually cannot balance the accuracy and efficiency well. So in this work, inspired by FCOS and GFL, authors propose an improved mobile-friendly and high-accuracy anchor-free detector named PP-PicoDet. To summarize, the main contributions are as follows:

* the CSP structure is adopted to construct CSP-PAN as the neck. The CSP-PAN unifies the input channel numbers by $1 × 1$ convolution for all branches of the neck, which significantly enhances the feature extraction ability and reduces network parameters. The $3 × 3$ depthwise separable convolution is enlarged to $5 × 5$ depthwise separable convolution to expand the receptive field.

* The label assignment strategy is essential in object detection. Authors use SimOTA (YOLOX) dynamic label assignment strategy and optimize some calculation details. Specifically, they use the weighted sum of Varifocal Loss (VFL) and GIoU loss to calculate the cost matrix, enhancing accuracy without harming efficiency.

* ShuffleNetV2 is cost-effective on mobile devices. Authors further enhance the network structure and propose a new backbone, namely Enhanced ShuffleNet (ES-Net), which performs better than ShuffleNetV2.

* An improved detection One-Shot Neural Architecture Search (NAS) pipeline is proposed to find the optimal architecture automatically for object detection. Authors straightly train the supernet on detection datasets, which leads to significant computational savings and optimization for detection. The NAS-generated models achieve better efficiency and accuracy trade-offs.

Full PP-PicoDet architecture:


<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/43f7e8d1-48df-4ace-98d8-2c199ec725c4" alt="pp_picodet_arch" height="400"/>
</p>


## Better Backbone

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/7a4f9e5d-4430-43f9-9679-a0801b46e3c5" alt="pp_picodet_es_block" height="400"/>
</p>

**Manually Designed Backbone**. ShuffleNetV2 is more robust than other networks on mobile devices. To further improve the performance of ShuffleNetV2, authors follow some methods of PP-LCNet to enhance the network structure and build a new backbone, namely **Enhanced ShuffleNet (ESNet)**. Figure above describes the ES Block of ESNet in detail. The SE module does a good job of weighting the network channels for better features. Therefore, the SE modules are added to all blocks. Like MobileNetV3, the activation functions for the two layers of the SE module are ReLU and H-Sigmoid, respectively. Channel shuffle provides the information exchange of ShuffleNetV2 channels, but it causes the loss of fusion features. To address this problem, authors add depthwise convolution and pointwise convolution to integrate different channel information when the stride is 2 (figure above (a)). GhostNet proposes a novel Ghost module that can generate more feature maps with fewer parameters to improve the network’s learning ability. PP-PicoDet adds the Ghost module in the blocks with stride set to 1 to further enhance the performance of our ESNet (figure above (b)).

**Neural Architecture Search**. Object detectors, equipped with high-performance backbones for classification, might be sub-optimal due to the gap between different tasks. Authors train and search the detection super-net directly on the detection datasets, which leads to significant computational savings and optimization of detection instead of classification. The framework simply consists of two steps:

1. Training the one-shot supernet on detection datasets
2. Architecture search on the trained supernet with an evolutionary algorithm (EA).

For convenience, authors simply use channel-wise search for backbone here. Specifically, they give flexible ratio options to choose different channel ratios. Ratio is randomly chosen in $[0.5, 0.675, 0.75, 0.875, 1]$. For example, 0.5 represents that the width is scaled by 0.5 of the full model. The channel numbers divisible by 8 can improve the speed of inference time on hardware devices. Therefore, instead of using the channel numbers in the original model, authors first trained the full model with channel numbers $[128, 256, 512]$ for each stage block. All ratio options can also keep the channel numbers divisible by 8. The chosen ratio works on all prunable convolutions in each block. All output channels are fixed as the full model. To avoid tedious hyper-parameter tuning, authors fix all original settings in the architecture search. For the training strategy, the sandwich rule is adopted to sample the largest (full) and the smallest child model and six randomly sampled child models for each training iteration. There are no more additional techniques adopted in the training strategy, such as distillation, since different techniques perform inconsistently for different models, especially for detection tasks. Finally, the selected architectures are retrained on ImageNet dataset and then trained on COCO.

## CSP-PAN and Detector Head

The PAN structured is used in PP-PicoDet to obtain multi-level feature maps and the CSP structure for feature concatenation and fusion between the adjacent feature maps. The CSP structure is widely used in the neck of YOLOv4 and YOLOX. In the original CSP-PAN, the channel number in each output feature map is kept the same as the input from the backbone. The structures with large channel numbers have expensive computational costs for mobile devices. Authors address this problem through making all channel numbers in all feature maps equal to the smallest channel number by $1 × 1$ convolution, as shown in full architecture figure. Top-down and bottom-up feature fusion are then used through the CSP structure. The scaled-down features lead to lower computation costs and undamaged accuracy. Furthermore, authors add a feature map scale to the top of CSP-PAN to detect more objects. At the same time, all convolutions except $1 × 1$ convolutions are depthwise separable convolution. Depthwise separable convolution expands the receptive field through $5 × 5$ convolution. This structure brings a considerable increase in accuracy with much fewer parameters. The specific structure is shown in the full architecture figure. In the detector head, PP-PicoDet uses depthwise separable convolution and $5 × 5$ convolution to expand the receptive field. The numbers of depthwise separable convolution can be set to 2, 4, or more. Both the neck and the head have four scale branches. Authors keep the channel numbers in the head consistent with the neck module and couple the classification and regression branches. YOLOX uses a decoupled head with fewer channel numbers to improve accuracy. PP-PicoDet's coupled head performs better without reducing the channel numbers. The parameters and the inference speed are almost the same as the decoupled head.

## Label Assignment Strategy and Loss

The label assignment of positive and negative samples has an essential impact on the object detectors. Most object detectors use fixed label assignment strategies. These strategies are straightforward. RetinaNet directly divides positive and negative samples by the IoU of the anchor and ground truth. FCOS takes the anchors whose center point is inside the ground truth as positive samples, and YOLOv4 and YOLOv5 select the location of the ground truth center point and its adjacent anchors as positive samples. ATSS determines the positive and negative samples based on the statistical characteristics of the nearest anchors around the ground truth. The abovementioned label assignment strategies are immutable in the global training process. SimOTA is a label assignment strat egy that changes continuously with the training process and achieves good results in YOLOX. PP-PicoDet uses SimOTA dynamic label assignment strategy to optimize the training process. SimOTA first determines the candidate area through the center prior, and then calculates the IoU of the predicted box and ground truth in the candidate area, and finally obtains parameter $κ$ by summing the $n$ largest IoU for each ground truth. The cost matrix is obtained by directly calculating the loss for all predicted boxes and ground truth in the candidate area. For each ground truth, the anchors corresponding to the smallest $κ$ loss are selected and assigned as positive samples. The original SimOTA uses the weighted sum of CE loss and IoU loss to calculate the cost matrix. To align the cost in SimOTA and the objective function, PP-PicoDet uses the weighted sum of Varifocal loss and GIoU loss for cost matrix. The weight of GIoU loss is the $λ$, which is set to 6 as shown to best through the experiments. The specific formula is:

$$ cost = loss_{vfl} + λ · loss_{giou} $$

In the detector head, for classification, PP-PicoDet uses Varifocal loss to couple classification prediction and quality prediction. For regression, authors use GIoU loss and Distribution Focal Loss. The formula is as follows:

$$ loss = loss_{vfl} + 2 · loss_{giou} + 0.25 · loss_{dfl} $$

In all the above formulas, $loss_{vfl}$ means Varifocal Loss, $loss_{giou}$ means GIoU loss, $loss_{dfl}$ means Distribution Focal Loss.

## Other Strategies

In recent years, more activation functions have emerged that go beyond ReLU. Among these activation functions, H-Swish, a simplified version of the Swish activation function, is faster to compute and more mobile-friendly. PP-PicoDet replaces the activation function in detectors from ReLU to H-Swish. The performance improves significantly while keeping the inference time unchanged.

Different from linear step learning rate decay, cosine learning rate decay is exponentially decaying the learning rate. Cosine learning rate drops smoothly, which benefits the training process, especially when the batch size is large. Too much data augmentation always increases the regularization effect and makes the training more difficult to converge for lightweight models. So in this work, authors only use random flip, random crop and multi-scale resize for data augmentation in training.
