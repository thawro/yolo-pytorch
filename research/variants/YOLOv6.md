# YOLO v6

2022 | [paper](https://arxiv.org/pdf/2209.02976) | _YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications_

In this report, authors empirically observed several important factors that motivate to refurnish the YOLO framework: 1) Reparameterization (and its scaling), 2) Quantization of reparameterization-based detectors, 3) Deployment on customer GPU, 4) Advanced domain-specific strategies like label assignment and loss function. The main contributions of YOLOv6 as follows:

* Refashion a line of networks of different sizes tailored for industrial applications in diverse scenarios
* Update YOLOv6 with a self-distillation strategy, performed both on the classification task and the regression task
* Broadly verify the advanced detection techniques for label assignment, loss function and data augmentation techniques and adopt them selectively to further boost the performance.
* Reform the quantization scheme for detection with the help of RepOptimizer and channel-wise distillation

The renovated design of YOLOv6 consists of the following components: **_network design_**, **_label assignment_**, **_loss function_**, **_data augmentation_**, **_industry-handy improvements_**, and **_quantization and deployment_**

YOLOv6

## Network Design

YOLOv6

A one-stage object detector is generally composed of the following parts: a **_backbone_**, a **_neck_** and a **_head_**:

* **Backbone** -  mainly determines the feature representation ability, meanwhile, its design has a critical influence on the inference efficiency since it carries a large portion of computation cost
* **Neck** - used to aggregate the low-level physical features with high-level semantic features, and then build up pyramid feature maps at all levels.
* **Head** - consists of several convolutional layers, and it predicts final detection results according to multi-level features assembled by the neck. It can be categorized as _anchor-based_ and _anchor-free_, or rather _parameter-coupled_ head and _parameter-decoupled_ head from the structure’s per- spective.

In YOLOv6, based on the principle of hardware-friendly network design, authors propose two scaled re-parameterizable backbones and necks to accommodate models at different sizes, as well as an efficient decoupled head with the hybrid-channel strategy. The overall architecture of YOLOv6 is shown in figure above.

**Backbone**: Compared with other mainstream architectures, RepVGG backbones are equipped with more feature representation power in small networks at a similar inference speed, whereas it can hardly be scaled to obtain larger models due to the explosive growth of the parameters and computational costs. In this regard, authors take RepBlock as the building block of the small networks and a more efficient CSPStackRep Block (CSP-based) for large models.\
**Neck**: The neck of YOLOv6 adopts PAN topology following YOLOv4 and YOLOv5. Authors enhance the neck with RepBlocks or CSPStackRep Blocks to have Rep- PAN.\
**Head**: Authors simplify the decoupled head to make it more efficient, called Efficient Decoupled Head.

yolov6_RepBlock

### Backbone

As mentioned above, the design of the backbone network has a great impact on the effectiveness and efficiency of the detection model. Previously, it has been shown that multi-branch networks can often achieve better classification performance than single-path ones, but often it comes with the reduction of the parallelism and results in an increase of inference latency. On the contrary, plain single-path networks like VGG take the advantages of high parallelism and less memory footprint, leading to higher inference efficiency. Lately in RepVGG, a structural re-parameterization method is proposed to decouple the training-time multi-branch topology with an inference-time plain architecture to achieve a better speed-accuracy trade-off.\

Inspired by the above works, authors design an efficient re-parameterizable backbone denoted as **_EfficientRep_**. For small models, the main component of the backbone is _Rep-Block_ during the training phase, as shown in figure above (a). And each _RepBlock_ is converted to stacks of $3 × 3$ convolu- tional layers (denoted as _RepConv_) with ReLU activation functions during the inference phase, as shown in (b). Typically a $3 × 3$ convolution is highly optimized on mainstream GPUs and CPUs and it enjoys higher computational density. Consequently, _EfficientRep_ Backbone sufficiently utilizes the computing power of the hardware, resulting in a significant decrease in inference latency while enhancing the representation ability in the meantime.\

However, authors notice that with the model capacity further expanded, the computation cost and the number of parameters in the single-path plain network grow exponentially. To achieve a better trade-off between the computation burden and accuracy, they revise a **_CSPStackRep_** Block to build the backbone of medium and large networks. As shown in (c), _CSPStackRep_ Block is composed of three $1 × 1$ convolution layers and a stack of sub-blocks consisting of two _RepVGG_ blocks or _RepConv_ (at training or inference respectively) with a residual connection. Besides, a cross stage partial (_CSP_) connection is adopted to boost performance without excessive computation cost. Compared with _CSPRepResStage_ (from PP-YOLOE), it comes with a more succinct outlook and considers the balance between accuracy and speed.

### Neck

In practice, the feature integration at multiple scales has been proved to be a critical and effective part of object detection. Authors adopt the modified PAN topology from YOLOv4 and YOLOv5 as the base of the detection neck. In addition, they replace the _CSP-Block_ used in YOLOv5 with _RepBlock_ (for small models) or _CSPStackRep_ Block (for large models) and adjust the width and depth accordingly. The neck of YOLOv6 is denoted as **_Rep-PAN_**.

### Head

**Efficient decoupled head**. The detection head of YOLOv5 is a coupled head with parameters shared between the classification and localization branches, while its counterparts in FCOS and YOLOX decouple the two branches, and additional two $3 × 3$ convolutional layers are introduced in each branch to boost the performance. In YOLOv6, authors adopt a hybrid-channel strategy to build a more efficient decoupled head. Specifically, they reduce the number of the middle $3 × 3$ convolutional layers to only one. The width of the head is jointly scaled by the width multiplier for the backbone and the neck. These modifications further reduce computation costs to achieve a lower inference latency.

**Anchor-free**. _Anchor-free_ detectors stand out because of their better generalization ability and simplicity in decoding prediction results. The time cost of its post-processing is substantially reduced. There are two types of _anchor-free_ detectors: anchor _point-based_ and _keypoint-based_. In YOLOv6, authors adopt the anchor _point-based_ paradigm, whose box regression branch actually predicts the distance from the anchor point to the four sides of the bounding boxes.

## Label Assignment

Authors evaluate the recent progress of label assignment strategies on YOLOv6 through numerous experiments, and the results indicate that TAL is more effective and training-friendly.

Label assignment is responsible for assigning labels to predefined anchors during the training stage. Previous work has proposed various label assignment strategies ranging from simple IoU-based strategy and inside ground-truth method to other more complex schemes.

**SimOTA**. OTA considers the label assignment in object detection as an Optimal Transmission problem. It defines positive/negative training samples for each ground-truth object from a global perspective. SimOTA is a simplified version of OTA, which reduces additional hyperparameters and maintains the performance. SimOTA was utilized as the label assignment method in the early version of YOLOv6. However, in practice, authors find that introducing SimOTA slows down the training process and it is not rare to fall into unstable training. Therefore, they desire a replacement for SimOTA, which is TAL.

**Task Alignment Learning**. Task Alignment Learning (TAL) was first proposed in TOOD, in which a unified metric of classification score and predicted box quality is designed. The IoU is replaced by this metric to assign object labels. To a certain extent, the problem of the misalignment of tasks (classification and box regression) is alleviated. The other main contribution of TOOD is about the task-aligned head (T-head). T-head stacks convolutional layers to build interactive features, on top of which the Task-Aligned Predictor (TAP) is used. PP-YOLOE improved T-head by replacing the layer attention in T-head with the lightweight ESE attention, forming ET-head. However, YOLOv6 authors find that the ET-head deteriorates the inference speed in their models and it comes with no accuracy gain. Therefore they retain the design of the Efficient decoupled head. Furthermore, authors observed that TAL could bring more performance improvement than SimOTA and stabilize the training. Therefore, they adopt TAL as the default label as- signment strategy in YOLOv6.

## Loss function

The loss functions of the mainstream anchor-free object detectors contain _classification loss_, _box regression loss_ and _object loss_. For each loss, authors systematically experiment it with all available techniques and finally select VariFocal Loss as the classification loss and SIoU/GIoU Loss as the regression loss.

Object detection contains two sub-tasks: _classification_ and _localization_, corresponding to two loss functions: _classification loss_ and _box regression loss_. For each sub-task, there are various loss functions presented in recent years.

### Classification Loss

Improving the performance of the classifier is a crucial part of optimizing detectors. _Focal Loss_ modified the traditional cross-entropy loss to solve the problems of class imbalance either between positive and negative examples, or hard and easy samples. To tackle the inconsistent usage of the quality estimation and classification between training and inference, _Quality Focal Loss_ (QFL) further extended _Focal Loss_ with a joint representation of the classification score and the localization quality for the supervision in classification. Whereas _VariFocal Loss_ (VFL) is rooted from _Focal Loss_, but it treats the positive and negative samples asymmetrically. By considering positive and negative samples at different degrees of importance, it balances learning signals from both samples. _Poly Loss_ decomposes the commonly used classification loss into a series of weighted polynomial bases. It tunes polynomial coefficients on different tasks and datasets, which is proved better than _Cross-entropy Loss_ and _Focal Loss_ through experiments. Authors of YOLOv6 assess all these advanced classification losses on YOLOv6 to finally adopt **_VFL_**.

### Box Regression Loss

Box regression loss provides significant learning signals localizing bounding boxes precisely. $L_1$-loss is the original box regression loss in early works. Progressively, a variety of well-designed box regression losses have sprung up, such as IoU-series loss and probability loss.

**IoU-series Loss**. IoU loss regresses the four bounds of a predicted box as a whole unit. It has been proved to be effective because of its consistency with the evaluation metric. There are many variants of IoU, such as _GIoU_, _DIoU_, _CIoU_, _α-IoU_ and _SIoU_, etc, forming relevant loss functions. Authors of YOLOv6 experiment with GIoU, CIoU and SIoU in this work. Finally _SIoU_ is applied to YOLOv6-N and YOLOv6-T, while others use _GIoU_.

**Probability Loss**. _Distribution Focal Loss_ (DFL) simplifies the underlying continuous distribution of box locations as a discretized probability distribution. It considers ambiguity and uncertainty in data without introducing any other strong priors, which is helpful to improve the box lo- calization accuracy especially when the boundaries of the ground-truth boxes are blurred. Upon _DFL_, _DFLv2_ develops a lightweight sub-network to leverage the close correlation between distribution statistics and the real localization quality, which further boosts the detection performance. However, _DFL_ usually outputs 17× more regression values than general box regression, leading to a substantial overhead. The extra computation cost significantly hinders the training of small models. Whilst _DFLv2_ further increases the computation burden because of the extra sub-network. In YOLOv6 experiments, _DFLv2_ brings similar performance gain to _DFL_. Consequently, authors only adopt _DFL_ in YOLOv6-M/L.

### Object Loss

Object loss was first proposed in _FCOS_ to reduce the score of low-quality bounding boxes so that they can be filtered out in post-processing. It was also used in YOLOX to accelerate convergence and improve network accuracy. As an anchor-free framework like _FCOS_ and YOLOX, authors have tried object loss into YOLOv6. Unfortunately, it doesn’t bring many positive effects

## Data Augmentation

## Industry improvements

Authors introduce additional common practice and tricks to improve the performance including self-distillation and more training epochs. For self-distillation, both classification and box regression are respectively supervised by the teacher model. The distillation of box regression is made possible thanks to DFL. In addition, the proportion of information from the soft and hard labels is dynamically declined via cosine decay, which helps the student selectively acquire knowledge at different phases during the training process. In addition, authors encounter the problem of the impaired performance without adding extra gray borders at evaluation, for which they provide some remedies.

The following tricks come ready to use in real practice. They are not intended for a fair comparison but steadily produce performance gain without much tedious effort: 1) More training epochs, 2) Self-distillation, 3) Gray border of images.

### More training epochs

Empirical results have shown that detectors have a progressing performance with more training time. We extended the training duration from 300 epochs to 400 epochs to reach a better convergence

### Self-distillation

To further improve the model accuracy while not introducing much additional computation cost, authors apply the classical knowledge distillation technique minimizing the _KL-divergence_ between the prediction of the teacher and the student. They limit the teacher to be the student itself but pretrained, hence it is called _self-distillation_. Note that the _KL-divergence_ is generally utilized to measure the difference between data distributions. However, there are two sub-tasks in object detection, in which only the classification task can directly utilize knowledge distillation based on _KL-divergence_. Thanks to _DFL_ loss, it is possible to perform it on box regression as well. The knowledge distillation loss can then be formulated as:

$$ L_{KD} = KL(p_{t}^{cls} ||p_s^{cls}) + KL(p_{t}^{reg} ||p_s^{reg}) $$

where $p_{t}^{cls}$ and $p_s^{cls}$ are class prediction of the teacher model and the student model respectively, and accordingly $p_{t}^{reg}$ and $p_{s}^{reg}$ are box regression predictions. The overall loss function is now formulated as:

$$ L_{total} = L_{det} + α LKD $$

where $L_{det}$ is the detection loss computed with predictions and labels. The hyperparameter $α$ is introduced to balance two losses. In the early stage of training, the soft labels from the teacher are easier to learn. As the training continues, the performance of the student will match the teacher so that the hard labels will help student more. Upon this, authors apply cosine weight decay to $α$ to dynamically adjust the information from hard labels and soft ones from the teacher.

### Gray border of images

Authors of YOLOv6 notice that a half-stride gray border is put around each image when evaluating the model performance in the implementations of YOLOv5 and YOLOv7. Although no useful information is added, it helps in detecting the objects near the edge of the image. This trick also applies in YOLOv6.\
However, the extra gray pixels evidently reduce the inference speed. Without the gray border, the performance deteriorates. The problem is related to the gray borders padding in _Mosaic_ augmentation. Experiments on turning _mosaic_ augmentations off during last epochs (aka. _fade strategy_) are conducted for verification. In this regard, authors change the area of gray border and resize the image with gray borders directly to the target image size. Combining these two strategies, models can maintain or even boost the performance without the degradation of inference speed.

## Quantization

To cure the performance degradation in quantizing reparameterization-based models, authors train YOLOv6 with RepOptimizer to obtain PTQ-friendly (Post Training Quantization) weights. They further adopt QAT (Quantization Aware Training) with channel-wise distillation and graph optimization to pursue extreme performance. The quantized YOLOv6-S hits a new state of the art with 42.3% AP and a throughput of 869 FPS (batch size=32).

For industrial deployment, it has been common practice to adopt quantization to further speed up runtime without much performance compromise. Post-Training Quantization (PTQ) directly quantizes the model with only a small calibration set. Whereas Quantization-Aware Training (QAT) further improves the performance with the access to the training set, which is typically used jointly with distillation. However, due to the heavy use of re-parameterization blocks in YOLOv6, previous PTQ techniques fail to produce high performance, while it is hard to incorporate QAT when it comes to matching fake quantizers during training and inference. Authors demonstrate the pitfalls and their cures during deployment.

### Reparameterizing Optimizer

yolov6_quantization_weights

RepOptimizer proposes gradient re-parameterization at each optimization step. This technique also well solves the quantization problem of reparameterization-based models. Authors hence reconstruct the re-parameterization blocks of YOLOv6 in this fashion and train it with RepOptimizer to obtain PTQ-friendly weights. The distribution of feature map is largely narrowed (see figure above (bottom)), which greatly benefits the quantization process.

### Sensitivity Analysis

Authors further improve the PTQ performance by partially converting quantization-sensitive operations into float computation. To obtain the sensitivity distribution, several metrics are commonly used, mean-square error (MSE), signal-noise ratio (SNR) and cosine similarity. Typically for comparison, one can pick the output feature map (after the activation of a certain layer) to calculate these metrics with and without quantization. As an alternative, it is also viable to compute validation AP by switching quantization on and off for the certain layer. Authors compute all these metrics on the YOLOv6-S model trained with RepOptimizer and pick the top-6 sensitive layers to run in float.

### Quantization-aware Training with Channel-wise Distillation

yolov6_channel_wise_distilation

In case PTQ is insufficient, authors propose to involve Quantization-Aware Training (QAT) to boost quantization performance. To resolve the problem of the inconsistency of fake quantizers during training and inference, it is necessary to build QAT upon the RepOptimizer. Besides, channel-wise distillation is adapted within the YOLOv6 framework, shown in figure above. This is also a self-distillation approach where the teacher network is the student itself in FP32-precision.
