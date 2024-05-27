# PP-YOLOE

2022 | [paper](http://arxiv.org/pdf/2203.16250v3) | _PP-YOLOE: An evolved version of YOLO_

In this report, authors present PP-YOLOE, an industrial SoTA object detector with high performance and friendly deployment. They optimize on the basis of the previous PP-YOLOv2, using anchor-free paradigm, more powerful backbone and neck equipped with **_CSPRepResStage_**, **_ET-head_** and dynamic label assignment algorithm **_TAL_**.

Based on PP-YOLOv2, authors proposed an evolved version of YOLO named PP-YOLOE. PP-YOLOE avoids using operators like deformable convolution and Matrix NMS to be well supported on various hardware. Moreover, PP-YOLOE can easily scale to a series of models for various hardware with different computing power. These characteristics further promote the application of PP-YOLOE in a wider range of practical scenarios.

## A Brief Review of PP-YOLOv2

The overall architecture of PP-YOLOv2 contains:

* [ResNet50-vd](https://arxiv.org/pdf/1812.01187) backbone with deformable convolution
* PAN neck with SPP layer and DropBlock
* the lightweight IoU aware head
* ReLU activation function is used in backbone while mish activation function is used in neck
* Following YOLOv3, PP-YOLOv2 only assigns one anchor box for each ground truth object. 
* In addition to classification loss, regression loss and objectness loss, PP-YOLOv2 also uses IoU loss and IoU aware loss to boost the performance

## Improvement of PP-YOLOE

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f0e4c598-9f64-4c34-8c9b-f377d4d16843" alt="pp_yolo_e" height="350"/>
</p>

### Anchor-free

As mentioned above, PP-YOLOv2 assigns ground truths in an anchor-based manner. However, anchor mechanism introduces a number of hyper-parameters and depends on hand-crafted design which may not generalize well on other datasets. For the above reason, PP-YOLOE introduces anchor-free method in PP-YOLOv2. Following FCOS, which tiles one anchor point on each pixel, authors set upper and lower bounds for three detection heads to assign ground truths to corresponding feature map. Then, the center of bounding box is calculated to select the closest pixel as positive samples. Following YOLO series, a $4D$ vector $(x, y, w, h)$ is predicted for regression. This modification makes the model a little faster with the loss of 0.3 AP. Although upper and lower bounds are carefully set according to the anchor sizes of PP-YOLOv2, there are still some minor inconsistencies in the assignment results between anchor-based and anchor-free manner, which may lead to little precision drop.

### Backbone and Neck

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/46d349c1-123b-4e2f-aba4-0003508f71ff" alt="pp_yolo_e_RepResBlock" height="300"/>
</p>

Residual connections and dense connections have been widely used in modern convolutional neural network. Residual connections introduce shortcut to relieve gradient vanishing problem and can be also regarded as a model ensemble approach. Dense connections aggregate intermediate features with diverse receptive fields, showing good performance on the object detection task. CSPNet utilizes cross stage dense connections to lower computation burden without the loss of precision, which is popular among effective object detectors such as YOLOv5, YOLOX. VoVNet and subsequent TreeNet also show superior performance in object detection and instance segmentation. Inspired by these works, PP-YOLOE proposed a novel **RepResBlock** by combining the residual connections and dense connections, which is used in the backbone and neck.
Originating from TreeBlock, the RepResBlock is shown in figure 2b during the training phase and 2c during the inference phase. Firstly, authors simplify the original TreeBlock (figure 2a). Then, they replace the concatenation operation with element-wise add operation (figure 2b), because of the approximation of these two operations to some extent shown in RMNet. Thus, during the inference phase, it is possible to re-parameterize RepResBlock to a basic residual block (figure 2c) used by ResNet-34 in a RepVGG style. 
PP-YOLOE uses proposed RepResBlock to build backbone and neck. Similar to ResNet, the backbone, named **CSPRepResNet**, contains one stem composed of three convolution layer and four subsequent stages stacked by the RepResBlock as shown in figure 3d. In each stage, cross stage partial (CSP) connections are used to avoid numerous parameters and computation burden brought by lots of $3 × 3$ convolution layers. **ESE** (Effective Squeeze and Extraction) layer is also used to impose channel attention in each CSPRepResStage while building backbone. Authors build neck with proposed RepResBlock and CSPRepResStage following PP-YOLOv2. Different from backbone, shortcut in RepResBlock and ESE layer in CSPRepResStage are removed in neck.
Authors use width multiplier $α$ and depth multiplier $β$ to scale the basic backbone and neck jointly like YOLOv5. Thus, they can get a series of detection network with different parameters and computation cost. The width setting of basic backbone is $[64, 128, 256, 512, 1024]$. Except for the _stem_, the depth setting of basic backbone is $[3, 6, 6, 3]$. The width setting and depth setting of basic neck are $[192, 384, 768]$ and $3$ respectively.

### Task Alignment Learning (TAL)

To further improve the accuracy, label assignment is another aspect to be considered. YOLOX uses SimOTA as the label assignment strategy to improve performance. However, to further overcome the misalignment of classification and localization, task alignment learning (TAL) is proposed in TOOD, which is composed of a dynamic label assignment and task aligned loss. Dynamic label assignment means prediction/loss aware. According to the prediction, it allocates dynamic number of positive anchors for each ground-truth. By explicitly aligning the two tasks, TAL can obtain the highest classification score and the most precise bounding box at the same time. For task aligned loss, TOOD use a normalized $t$, namely $\hat{t}$, to replace the target in loss. It adopts the largest IoU within each instance as the normalization. The Binary Cross Entropy (BCE) for the classification can be rewritten as:

$$ L_{cls−pos} = \sum_{i=1}^{N_pos} BCE (p_i, \hat{t}_i) $$

Authors investigate the performance using different label assignment strategies and conduct the experiment on above modified model, which uses CSPRepResNet as backbone. The models were trained for 36 epochs only (on COCO train2017) and validated on COCO val. TAL achieves the best 45.2% AP performance, so PP-YOLOE uses TAL.

### Efficient Task-aligned Head (ET-head)

In object detection, the task conflict between classification and localization is a well-known problem. Corresponding solutions are proposed in many papers. YOLOX’s decoupled head draws lessons from most of the one-stage and two-stage detectors, and successfully applies it to YOLO model to improve accuracy. However, the decoupled head may make the classification and localization tasks separate and independent, and lack of task specific learning. Based on TOOD, PP-YOLOE improves the head and propose **ET-head** with the goal of both speed and accuracy. As shown in first figure, PP-YOLOE uses **ESE** to replace the layer attention in TOOD, simplify the alignment of classification branches to shortcut, and replace the alignment of regression branches with **Distribution Focal Loss (DFL)** layer. Through the above changes, the ET-head brings an increase of 0.9ms on V100. For the learning of classification and location tasks, authors choose **Varifocal Loss (VFL)** and **Distribution Focal Loss (DFL)** respectively. PP-Picodet successfully applied VFL and DFL in object detectors, and obtains performance improvement. VFL (different from the **Quality Focal Loss (QFL)**) uses target score to weight the loss of positive samples. This implementation makes the contribution of positive samples with high IoU to loss relatively large. This also makes the model pay more attention to high-quality samples rather than those low-quality ones at training time. The same is that both use IoU-aware classification score (IACS) as the target to predict. This can effectively learn a joint representation of classification score and localization quality estimation, which enables high consistency between training and inference. For DFL, in order to solve the problem of inflexible representation of bounding box, GFL proposes to use general distribution to predict bounding box. Our model is supervised by the following loss function:

$$ Loss = \frac{α · loss_{VFL} + β · loss_{GIoU} + γ · loss_{DFL}}{\sum_{i}^{N_{pos}} \hat{t}}$$

where $\hat{t}$ denotes the normalized target score.
