# RTMDet

2022 | [paper](http://arxiv.org/pdf/2212.07784v2) | _RTMDet: An Empirical Study of Designing Real-Time Object Detectors_

In this paper, authors aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, they explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel _depth-wise convolutions_. Authors further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named **_RTMDet_** (**R**eal-**T**ime **M**odels for object **Det**ection), achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet also obtains new SoTA performance on real-time instance segmentation and rotated object detection. The appealing improvements mainly come from better representation with large-kernel depth-wise convolutions and better optimization with soft labels in the dynamic label assignments.

Specifically, authors first exploit large-kernel _depth-wise convolutions_ in the basic building block of the backbone and neck in the model, which improves the model’s capability of capturing the global context. Because directly placing depth-wise convolution in the building block will increase the model depth thus slowing the inference speed, they further reduce the number of building blocks to reduce the model depth and compensate for the model capacity by increasing the model width. We also observe that putting more parameters in the neck and making its capacity compatible with the backbone could achieve a better speed-accuracy trade-off. The overall modification of the model architectures allows the fast inference speed of RTMDet without relying on model re-parameterizations.

Authors further revisit the training strategies to improve the model accuracy. In addition to a better combination of data augmentations, optimization, and training schedules, they empirically find that existing dynamic label assignment strategies can be further improved by introducing soft targets instead of hard labels when matching ground truth boxes and model predictions. Such a design improves the discrimination of the cost matrix for high-quality matching but also reduces the noise of label assignment, thus improving the model accuracy.

The macro architecture of RTMDet is a typical one-stage object detector (see figure below). Authors improve the model efficiency by:

1. Exploring the large-kernel convolutions in the basic building block of backbone and neck
2. Balancing the model depth, width, and resolution accordingly
3. Exploring soft labels in dynamic label assignment strategies and a better combination of data augmentations and optimization strategies to improve the model accuracy
4. Extending RTMDet to instance segmentation and rotated object detection tasks with few modifications

rtmdet_overview
<p align="center">
  <img src="" alt="" height="300"/>
</p>

**Macro architecture**. Authors decompose the macro architecture of a one-stage object detector into the **_backbone_**, **_neck_**, and **_head_**, as shown in figure above. Recent advances of YOLO series typically adopt CSPDarkNet as the backbone architecture, which contains four stages and each stage is stacked with several basic building blocks (figure below (a)). The neck takes the multi-scale feature pyramid from the backbone and uses the same basic building blocks as the backbone with bottom-up and top-down feature propogation to enhance the pyramid feature map. Finally, the detection head predicts object bounding boxes and their categories based on the feature map of each scale. Such an architecture generally applies to general and rotated objects, and can be extended for instance segmentation by the kernel and mask feature generation heads (from [paper](https://arxiv.org/pdf/2003.05664)).

To fully exploit the potential of the macro architecture, authors first study more powerful basic building blocks, then investigate the computation bottleneck in the architecture and balance the depth, width, and resolution in the backbone and neck.

## Model Architecture

### Basic building block

rtmdet_building_blocks
<p align="center">
  <img src="" alt="" height="300"/>
</p>

A large effective receptive field in the backbone is beneficial for dense prediction tasks like object detection and segmentation as it helps to capture and model the image context more comprehensively. However, previous attempts (e.g., dilated convolution and non-local blocks) are computationally expensive, limiting their practical use in real-time object detection. Recent studies revisit the use of large-kernel convolutions, showing that one can enlarge the receptive field with a reasonable computational cost through _depth-wise convolution_. Inspired by these findings, RTMDet introduces $5 × 5$ _depth-wise convolutions_ in the basic building block of CSPDarkNet to increase the effective receptive fields (figure above (b)). This approach allows for more comprehensive contextual modeling and significantly improves accuracy.

It is noteworthy that some recent real-time object detectors explore re-parameterized $3 × 3$ convolutions in the basic building block (figure above (c) and (d)). While the re-parameterized $3 × 3$ convolutions is considered a free lunch to improve accuracy during inference, it also brings side effects such as slower training speed and increased training memory. It also increases the error gap after the model is quantized to lower bits, requiring compensation through re-parameterizing optimizer (RepOptimizer) and Quantization-Aware Training. Large-kernel depth-wise convolution is a simpler and more effective option for the basic building block compared to re-parameterized $3 × 3$ convolution, as they require less training cost and cause less error gaps after model quantization.

### Balance of model width and depth

The number of layers in the basic block also increases due to the additional point-wise convolution following the large-kernel depthwise convolution (above figure (b)). This hinders the parallel computation of each layer and thus decreases inference speed. To address this issue, RTMDet reduces the number of blocks in each backbone stage and moderately enlarges the width of the block to increase the parallelization and maintain the model capacity, which eventually improves inference speed without sacrificing accuracy.

### Balance of backbone and neck

Multi-scale feature pyramid is essential for object detection to detect objects at various scales. To enhance the multi-scale features, previous approaches either use a larger backbone with more parameters or use a heavier neck with more connections and fusions among feature pyramid. However, these attempts also increase the computation and memory footprints. Therefore, RTMDet adopts another strategy that puts more parameters and computations from backbone to neck by increasing the expansion ratio of basic blocks in the neck to make them have similar capacities, which obtains a better computation-accuracy trade-off.

### Shared detection head

Real-time object detectors typically utilize separate detection heads for different feature scales to enhance the model capacity for higher performance, instead of sharing a detection head across multiple scales. Authors of RTMDet compared different design choices and choose to share parameters of heads across scales but incorporate different _Batch Normalization_ (BN) layers to reduce the parameter amount of the head while maintaining accuracy. BN is also more efficient than other normalization layers such as _Group Normalization_ because in inference it directly uses the statistics calculated in training.

## Training Strategy

### Label assignment and losses

To train the one-stage object detector, the dense predictions from each scale will be matched with ground truth bounding boxes through different label assignment strategies. Recent advances typically adopt dynamic label assignment strategies that use cost functions consistent with the training loss as the matching criterion. However, authors find that their cost calculation have some limitations. Hence, they propose a dynamic soft label assignment strategy based on SimOTA, and its cost function is formulated as:

$$ C = λ_1 C_{cls} + λ_2 C_{reg} + λ_3 C_{center} $$

where $C_{cls}$, $C_{reg}$ $C_{center}$ , and correspond to the classification cost, regression cost and region prior cost, respectively, and $λ_1 = 1$, $λ_2 = 3$, and $λ_3 = 1$ are the weights of these three costs by default.

**Classification cost ($C_{cls}$)**. Previous methods usually utilize binary labels to compute classification cost $C_{cls}$, which allows a prediction with a high classification score but an incorrect bounding box to achieve a low classification cost and vice versa. To solve this issue, RTMDet introduces soft labels in $C_{cls}$ as:

$$ C_{cls} = CE(P, Y_{soft}) × (Y_{soft} − P)^2 $$

**Regression cost ($C_{reg}$)**. The modification is inspired by GFL that uses the IoU between the predictions and ground truth boxes as the soft label $Y_{soft}$ to train the classification branch. The soft classification cost in assignment not only reweights the matching costs with different regression qualities but also avoids the noisy and unstable matching caused by binary labels. When using Generalized IoU as regression cost, the maximum difference between the best match and the worst match is less than 1. This makes it difficult to distinguish high-quality matches from low-quality matches. To make the match quality of different GT-prediction pairs more discriminative, RTMDet uses the logarithm of the IoU as the regression cost instead of GIoU used in the loss function, which amplifies the cost for matches with lower IoU values. The regression cost $C_{reg}$ is calculated by:

$$ C_{reg} = −log(IoU) $$

**Region cost ($C_{center}$)**. For region cost $C_{center}$, RTMDet uses a soft center region cost instead of a fixed center prior to stabilize the matching of the dynamic cost:

$$ C_{center} = α^{|x_{pred} − x_{gt}| − β} $$

where $α$ and $β$ are hyper-parameters of the soft center region. Authors set $α = 10$, $β = 3$ by default.

### Cached Mosaic and MixUp

Cross-sample augmentations such as _MixUp_ and _CutMix_ are widely adopted in recent object detectors. These augmentations are powerful but bring two side effects. First, at each iteration, they need to load multiple images to generate a training sample, which introduces more data loading costs and slows the training. Second, the generated training sample is "noisy" and may not belong to the real distribution of the dataset, which affects the model learning.

RTMDet improves _MixUp_ and _Mosaic_ with the caching mechanism that reduces the demand for data loading. By utilizing cache, the time cost of mixing images in the training pipeline can be significantly reduced to the level of processing a single image. The cache operation is controlled by the cache length and popping method:

* A large cache length and random popping method can be regarded as equivalent to the original non-cached _MixUp_ and _Mosaic_ operations
* A small cache length and First-In-First-Out (FIFO) popping method can be seen as similar to the repeated augmentation, allowing for the mixing of the same image with different data augmentation operations in the same or contiguous batches.

### Two-stage training

To reduce the side effects of "noisy" samples by strong data augmentations, YOLOX explored a two-stage training strategy, where the first stage uses strong data augmentations, including _Mosaic_, _MixUp_, and _Random Rotation_ and _Shear_, and the second stage use weak data augmentations, such as _Random Resizing_ and _Flipping_. As the strong augmentation in the initial training stage includes _Random Rotation_ and _Shearing_ that cause misalignment between inputs and the transformed box annotations, YOLOX adds the $L_1$-loss to finetune the regression branch in the second stage. To decouple the usage of data augmentation and loss functions, RTMDet **excludes** these data augmentations and increase the number of mixed images to 8 in each training sample in the first training stage of 280 epochs to compensate for the strength of data augmentation. In the last 20 epochs, RTMDet switches to _Large Scale Jittering_ (LSJ), allowing for finetuning of the model in a domain that is more closely aligned with the real data distributions. To further stabilize the training, RTMDet adopts _AdamW_ as the optimizer, which is rarely used in convolutional object detectors but is a default for vision transformers.

## Extending to other tasks

### Instance segmentation

rtmdet_instance
<p align="center">
  <img src="" alt="" height="400"/>
</p>

RTMDet can be used for instance segmentation with a simple modification, denoted as RTMDet-Ins. As illustrated in figure above, based on RTMDet, an additional branch is added, consisting of a kernel prediction head and a mask feature head, similar to [CondInst](https://arxiv.org/pdf/2003.05664). The mask feature head comprises 4 convolution layers that extract mask features with 8 channels from multi-level features. The kernel prediction head predicts a 169-dimensional vector for each instance, which is decomposed into three dynamic convolution kernels to generate instance segmentation masks through interaction with the mask features and coordinate features. To further exploit the prior information inherent in the mask annotations, authors use the mass center of the masks when calculating the soft region prior in the dynamic label assignment instead of the box center. RTMDet-Ins uses [Dice Loss](https://arxiv.org/pdf/1606.04797) as the supervision for the instance masks following typical conventions.

### Rotated object detection

Due to the inherent similarity between rotated object detection and general (horizontal) object detection, it only takes 3 steps to adapt RTMDet to a rotated object detector, noted as RTMDet-R:

1. Add a $1 × 1$ convolution layer at the regression branch to predict the rotation angle
2. Modify the bounding box coder to support rotated boxes
3. Replace the GIoU loss with Rotated IoU loss.

The highly optimized model architecture of RTMDet guarantees high performance of RTMDet-R on the rotated object detection tasks. Moreover, as RTMDet-R shares most parameters of RTMDet, the model weights of RTMDet pre-trained on general detection dataset (e.g., COCO) can serve as a good initialization for rotated object detection.
