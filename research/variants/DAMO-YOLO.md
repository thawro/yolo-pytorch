# DAMO-YOLO

2023 | [paper](http://arxiv.org/pdf/2211.15444v4) | _DAMO-YOLO : A Report on Real-Time Object Detection Design_

DAMO-YOLO is extended from YOLO with some new technologies, including Neural Architecture Search (NAS), efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. In particular, authors use MAE-NAS, a method guided by the principle of maximum entropy, to search the detection backbone under the constraints of low latency and high performance, producing ResNet-like / CSP-like structures with spatial pyramid pooling and focus modules. In the design of necks and heads, authors follow the rule of “large neck, small head”. They import Generalized-FPN with accelerated queen-fusion to build the detector neck and upgrade its CSPNet with efficient layer aggregation networks (ELAN) and reparameterization. Then authors investigate how detector head size affects detection performance and find that a heavy neck with only one task projection layer would yield better results. In addition, AlignedOTA is proposed to solve the misalignment problem in label assignment. And a distillation schema is introduced to improve performance to a higher level.

Recently, researchers have developed object detection methods in huge progress. While the industry pursues high-performance object detection methods with real-time constraints, researchers focus on designing one-stage detectors with efficient network architectures and advanced training stages. Especially, YOLOv5/6/7, YOLOX and PP-YOLOE have achieved significant AP-Latency trade-offs on COCO, making YOLO series object detection methods widely used in the industry.

Although object detection has achieved great progress, there are still new techs that can be brought in to further improve performance. Firstly, the network structure plays a critical role in object detection. Darknet holds a dominant position in the early stages of YOLO history. Recently, some works have investigated other efficient networks for their detectors, i.e., YOLOv6 and YOLOv7. However, these networks are still manually designed. Thanks to the development of the Neural Architecture Search (NAS), there are many detection-friendly network structures found through the NAS techs, which have shown great superiority over previous manually designed networks. Therefore, authors take advantage of the NAS techs and import MAE-NAS for the DAMO-YOLO. MAE-NAS is a heuristic and training-free neural architecture search method without supernet dependence and can be utilized to archive backbones at different scales. It can produce ResNet-like / CSP-like structures with spatial pyramid pooling and focus modules.

Secondly, it is crucial for a detector to learn sufficient fused information between high-level semantic and low-level spatial features, which makes the detector neck to be a vital part of the whole framework. The importance of neck has also been discussed in other works. Feature Pyramid Network (FPN) has been proved effective to fuse multi-scale features. Generalized-FPN (GFPN) improves FPN with a novel queen-fusion. In DAMO-YOLO, authors design a Reparameterized Generalized-FPN (**_RepGFPN_**). It is based on GFPN but the efficient layer aggregation networks (ELAN) and re-parameterization are involved in an accelerated queen-fusion.

To strike the balance between latency and performance, authors conducted a series of experiments to verify the importance of the neck and head of the detector and found that ”large neck, small head” would lead to better performance. Hence, they discard the detector head in previous YOLO series works and only leave a task projection layer. The saved calculations are moved to the neck part. Besides the task projection module, there is no other training layer in the head, so the detector head is called ZeroHead. Coupled with the **_RepGFPN_**, ZeroHead achieves SoTA performance.

In addition, the dynamic label assignment, such as OTA and TOOD, is widely acclaimed and achieves significant improvement compared to the static label assignment. However, the misalignment problem (between classification and regression) is still unsolved in these works. DAMO-YOLO proposes a better solution called **_AlignOTA_** to balance the importance of classification and regression, which can partly solve the problem.

At last, Knowledge Distillation (KD) has been proved effective in boosting small models by the larger model supervision. This tech does exactly fit the design of real-time object detection. Nevertheless, applying KD on YOLO series sometimes can not achieve significant improvements as hyperparameters are hard to optimize and features carry too much noise. In the DAMO-YOLO, authors first make distillation great again on models of all sizes, especially on small ones.

dama_yolo

<p align="center">
  <img src="" alt="" height="300"/>
</p>

## MAE-NAS Backbone

dama_yolo_blocks

<p align="center">
  <img src="" alt="" height="300"/>
</p>

Previously, in real-time scenarios, designers relied on the Flops-mAP curve as a simple means of assessing model performance. However, the relationship between a model’s flops and latency is not necessarily consistent. In order to enhance a model’s real-world performance in industrial deployment, DAMO-YOLO prioritized the latency-MAP curve in the design process. Based on this design principle, authors use MAE-NAS to obtain optimal networks under different latency budgets. MAE-NAS constructs an alternative proxy based on information theory to rank initialized networks without training. Therefore, the search process only takes a few hours, which is much lower than the training costs. Several basic search blocks are provided by MAE-NAS, such as Mob-block, Res-block and CSP-block, as shown in figure above. The Mob-block is a variant of MobileNetV3 block, the Res-block is derived from ResNet, and the CSP-block is derived from CSPNet.

Authors found that applying different kinds of blocks in models of different scales achieves better trade-offs for real-time inference. Authors compared the performance among CSP-Darknet and MAE-NAS backbones under our DAMO-YOLO with different scales. The MAE-CSP (apply CSP-block in the MAE-NAS backbones) obtained by MAE-NAS technology outperforms manually designed CSP-Darknet in terms of both speed and accuracy, demonstrating the superiority of MAE-NAS technology.  Moreover, using Res-block on smaller models can achieve a better trade-off between performance and speed than CSP-block, while using CSP-block can significantly outperform Res-block on larger and deeper networks. As a consequences, DAMO-YOLO uses "MAE-Res" in "T" (Tiny) and "S" models and "MAE-CSP" in "M" and "L" models in the final setting. When dealing with scenarios that only have limited computing power or GPU is not available, it is crucial to have models that meet strict requirements for computation and speed. To address this issue, authors have designed a series of light weight model using Mob-Block. Mob-block is derived from MobileNetV3, which can significantly decrease model computation and is friendly with CPU devices.

## Efficient RepGFPN

Feature pyramid network aims to aggregate different resolution features extracted from the backbone, which has been proven to be a critical and effective part of object detection. The conventional FPN introduces a top-down pathway to fuse multi-scale features. Considering the limitation of one-way information flow, PAFPN adds an additional bottom-up path aggregation network, but with higher computational costs. BiFPN removes nodes that only have one input edge, and adds skip- link from the original input on the same level. Generalized-FPN (GFPN) is proposed to serve as neck and achieves SoTA performance, as it can sufficiently exchange high-level semantic information and low-level spatial information. In GFPN, multi-scale features are fused in both level features in previous and current layers. What’s more, the $log_2(n)$ skip-layer connections provide more effective information transmission that can scale into deeper networks. When authors directly replaced modified-PANet with GFPN on modern YOLO-series models, they achieved higher precision. However, the latency of GFPN based model is much higher than modified-PANet-based model. By analyzing the structure of GFPN, the reason can be attributed to the following aspects:

1. Feature maps with different scales share the same dimension of channels
2. The operation of queen-fusion can not meet the requirement for real-time detection model
3. The convolution-based cross-scale feature fusion is not efficient.

Based on GFPN, authors propose a novel Efficient-RepGFPN to meet the design of real-time object detection, which mainly consists of the following insights:

1. Due to the large difference in FLOPs from different scale feature maps, it is difficult to control the same dimension of channels shared by each scale feature map under the constraint of limited computation cost. Therefore, in the feature fusion of DAMO-YOLO neck, authors adopt the setting of different scale feature maps with different dimensions of channels. By flexibly controlling the number of channels in different scales, it is possible to achieve much higher accuracy than sharing the same channels at all scales. Best performance is obtained when depth equals $3$ and width equals $(96, 192, 384)$
2. GFPN enhances feature interaction by queen-fusion, but it also brings lots of extra upsampling and downsampling operators. The benefits of those upsampling and downsampling operators are compared and authors note that the additional upsampling operator results in a latency increase of 0.6ms, while the accuracy improvement was only 0.3mAP, far less than the benefit of the additional downsampling operator. Therefore, under the constraints of real-time detection, they remove the extra upsampling operation in queen-fusion
3. In the feature fusion block, DAMO-YOLO first replaces original $3 x 3$ _conv_ based feature fusion with CSPNet and obtain 4.2 mAP gain. Afterward, they upgrade CSPNet by incorporating re-parameterization mechanism and connections of efficient layer aggregation networks (ELAN). Without bringing extra huge computation burden, DAMO-YOLO achieves much higher precision.

## ZeroHead

In recent advancements of object detection, decoupled head is widely used. With the decoupled head, those models achieve higher AP, while the latency grows significantly. To trade off the latency and the performance, authors have conducted a series of experiments to balance the importance of neck and head. From the experiments, authors find that "large neck, small head" would lead to better performance. Hence, they discard the decoupled head in previous works, but only left a task projection layer, i.e., one linear layer for classification and one linear layer for regression. Authors named this head **_ZeroHead_** as there is no other training layers in our head. ZeroHead can save computations for the heavy RepGFPN neck to the greatest extent. It is worth noticing that ZeroHead essentially can be considered as a coupled head, which is quite a difference from the decoupled heads in other works.

## Loss

In the loss after head, following GFocal, authors use Quality Focal Loss (QFL) for classification supervision, and Distribution Focal Loss (DFL) and GIOU loss for regression supervision. QFL encourages to learn a joint representation of classification and localization quality. DFL provides more informative and precise bounding box estimations by modeling their locations as General distributions. The training loss of the proposed DAMO-YOLO is formulated as:

$$ Loss = α loss_{QFL} + β loss_{DFL} + γ loss_{GIOU} $$

## AlignOTA

Besides head and loss, label assignment is a crucial component during detector training, which is responsible for assigning classification and regression targets to predefined anchors. Recently, dynamic label assignment such as OTA and TOOD is widely acclaimed and achieves significant improvements compares to static one. Dynamic label assignment methods assign labels according to the assignment cost between prediction and ground truth, e.g., OTA. Although the alignment of classification and regression in loss is widely studied, the alignment between classification and regression in label assignment is rarely mentioned in current works. The misalignment of classification and regression is a common issue in static assignment methods. Though dynamic assignment alleviates the problem, it still exists due to the unbalance of classification and regression losses, e.g., CrossEntropy and IoU Loss. To solve this problem, DAMO-YOLO introduces the focal loss into the classification cost, and use the IoU of prediction and ground truth box as the soft label, which is formulated as follows:

$$ AssignCost = C_{reg} + C_{cls} $$

$$ α = IoU (reg_{gt}, reg_{pred}) $$

$$ C_{reg} = −ln(α) $$

$$ C_{cls} = (α − cls_{pred})^2 × CE(cls_{pred}, α) $$

With this formulation, authors are able to choose the classification and regression aligned samples for each target. Besides the aligned assignment cost, following OTA, they form the solution of aligned assignment cost from a global perspective and name this label assignment as **_AlignOTA_**. AlignOTA outperforms all other label assignment methods.

## Distillation Enhancement

Knowledge Distillation (KD) is an effective method to further boost the performance of pocket-size models. Nevertheless, applying KD on YOLO series sometimes can not achieve significant improvements as hyperparameters are hard to optimize and features carry too much noise. In DAMO-YOLO, authors first make distillation great again on models of all sizes, especially on the small size. They adopt the feature-based distillation to transfer dark knowledge, which can distill both recognition and localization information in the intermediate feature maps. Authors conduct fast validation experiments to choose a suitable distillation method for the DAMO-YOLO and decide that CWD is the best fit for the models. The proposed distillation strategy is split into two stages:

1. The teacher distills the student at the first stage (284 epochs) on strong mosaic domain. Facing the challenging augmented data distribution, the student can further extract information smoothly under the teacher’s guidance
2. The student finetunes itself on no mosaic domain at the second stage (16 epochs). The reason why we do not adopt distillation at this stage is that, in such a short period, the teacher’s experience will damage the student’s performance when he wants to pull the student in a strange domain (i.e., no mosaic domain).

A long-term distillation would weaken the damage but is expensive. So authors choose a trade-off to make the student independent. In DAMO-YOLO, the distillation is equipped with two advanced enhancements:

1. **Align Module** - On the one hand, it is a linear projection layer to adapt student feature’s to the same resolution $(C, H, W)$ as teacher’s. On the other hand, forcing the student to approximate teacher feature directly leads to minor gains compared to the adaptive imitation
2. **Channel-wise Dynamic Temperature** - Inspired by PKD, authors add a normalization to teacher and student features, to weaken the effect the difference of real values brings. After subtracting the mean, standard deviation of each channel would function as temperature coefficient in KL loss.

Besides, authors present two key observations for a better usage of distillation. One is the balance between distillation and task loss. As shown in experiments, when one focuses more on distillation (weight=10), the classification loss in student has a slow convergence, which results in a negative effect. The small loss weight (weight=0.5) hence is necessary to strike a balance between distillation and classification. The other is the shallow head of detector. Authors found that the proper reduction of the head depth is beneficial to the feature distillation on neck. The reason is that when the gap between the final outputs and the distilled feature map is closer, distillation can have a better impact on decision. In practice, authors use DAMO-YOLO-S as teacher to distill DAMO-YOLO-T, and DAMO-YOLO-M as teacher to distill DAMO-YOLO-S, and DAMO-YOLO-L as teacher to distill DAMO-YOLO-M, while DAMO-YOLO-L is dis- tilled by it self.

## General Class DAMO-YOLO Model

General Class DAMO-YOLO model has been trained on multiple large-scale datasets, including COCO, Objects365, and OpenImage. The model incorporates several improvements that enable high precision and generalization. Firstly, authors address ambiguity resulting from overlapping categories across different datasets, such as "mouse" which can refer to a computer mouse in COCO/Objects365 and a rodent in OpenImage, by creating a unified label space for filtering. This reduces the original 945 categories (80 from COCO, 365 from Objects365, and 500 from OpenImage) to 701. Secondly, authors present a method of polynomial smoothing followed by weighted sampling to address the imbalance in dataset sizes, which often results in batch sampling biases towards larger datasets. Finally they implement a class-aware sampling approach to tackle the issue of long-tail effects in Objects365 and OpenImage datasets. This assigns greater sampling weights to images containing categories with fewer data points. By improving the model, they trained DAMO-YOLO-S on Large-Scale Datasets and achieved a 1.1% mAP improvement compared to the DAMO-YOLO-S baseline on COCO. Furthermore, this pre-trained model can be fine-tuned for downstream tasks.
