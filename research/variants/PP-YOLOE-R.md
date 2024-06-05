# PP-YOLOE-R

2022 | [paper](https://arxiv.org/pdf/2211.02386) | _PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector_

Compared with PP-YOLOE, the main changes of PP-YOLOE-R can be attributed to four aspects:

1. Introduced of [ProbIoU](https://arxiv.org/pdf/2106.06072) loss (like [FCOSR](https://arxiv.org/pdf/2111.10780)) as regression loss to avoid boundary discontinuity problem
2. Introduced of Rotated Task Alignment Learning to be suitable for rotated object detection on the basis of Task Alignment Learning
3. Designed a decoupled angle prediction head and directly learn the General distribution of angle through DFL loss for a more accurate angle prediction
4. A slight modification to the re-parameterization mechanism by adding a learnable gating unit to control the amount of information from the previous layer

pp_yolo_e_r

<p align="center">
  <img src="" alt="" height="300"/>
</p>

As shown in figure above the overall architecture of PP-YOLOE-R is similar to that of PP-YOLOE. PP-YOLOE-R improves detection performance of rotated bounding boxes at the expense of relatively small amount of parameters and computation based on PP-YOLOE. The changes made for rotated bounding boxes are introduced in the following sections.

pp_yolo_e_r_ablation

<p align="center">
  <img src="" alt="" height="300"/>
</p>

## Baseline

Drawing on FCOSR, authors introduced FCOSR Assigner and ProbIoU Loss into PP-YOLOE as the baseline. FCOSR Assigner is adopted to assign ground truth to three feature maps according to the predefined rules and ProbIoU Loss is utilized as regression loss. The backbone and neck of the baseline remain the same as PP-YOLOE while the regression branch of head is modified to predict five-parameter rotated bounding boxes $(x, y, w, h, θ)$ directly.

## Rotated Task Alignment Learning

Task Alignment Learning is composed of a task-aligned label assignment and task-aligned loss. Task-aligned label assignment constructs a task alignment metric to select positive samples from candidate anchor points, whose coordinates fall into any ground truth boxes. The task alignment metric is calculated as follows:

$$ t = s^α · μ^β $$

where $s$ denotes a predicted classification score and $μ$ denotes the IoU value between predicted bounding box and corresponding ground truth. In Rotated Task Alignment Learning, the selection process of candidate anchor points takes advantage of the geometric properties of the ground truth bounding box and anchor points in it and the SkewIoU value of predicted and ground truth bounding box is adopted as $μ$. With these two simple changes it is possible to apply task- aligned label assignment for rotated object detection and use task-aligned loss without modification.

## Decoupled Angle Prediction Head

In the regression branch, most rotated object detectors predict five parameters $(x, y, w, h, θ)$ to represent the oriented object. However, authors assume that predicting $θ$ requires different features than predicting $(x, y, w, h)$. To verify this hypothesis, they design a decoupled angle prediction head as shown in first figure to predict $θ$ and $(x, y, w, h)$ separately. The angle prediction head consists of a channel attention layer and a convolution layer and is very lightweight.

## Angle Prediction with DFL

ProbIoU Loss is adopted as regression loss to jointly optimize $(x, y, w, h, θ)$. To calculate ProbIoU Loss, the rotated bounding box is converted to a Gaussian bounding box. When the rotated bounding box is roughly square, the orientation of the rotated bounding box cannot be determined because the orientation in Gaus- sian bounding box is inherited from the elliptical representation. To overcome this problem, authors introduce Distribution Focal Loss (DFL) to predict the angle. Different form $l_n$-norm (which learns the Dirac delta distribution), DFL is aimed at learning the General distribution of angle. Specifically, authors discretize the angle with even intervals $ω$ and obtain the preidcted $θ$ in the form of integral, which can be formulated as follows:

$$ θ = \sum_{i=0}^{90} p_i · i · ω $$

where $p_i$ means the probability that the angle falls in each interval. In this paper, the rotated bounding box is defined in OpenCV definition and $ω$ is set to $\frac{π}{180}$.

## Learnable Gating Unit for RepVGG

RepVGG proposes a multi-branch architecture composed of a $3 × 3$ _conv_, a $1 × 1$ _conv_ and a shortcut path. The training-time information flow of RepVGG can be formulated as follows:

$$ y = f(x) + g(x) + x $$

while $f(x)$ is a $3 × 3$ _conv_ and $g(x)$ is a $1 × 1$ _conv_. During inference, it is possible to re-parameterizes this architecture to an equivalent $3 × 3$ _conv_. Although RepVGG is equivalent to a convolution layer, using RepVGG during training converges better. Authors attribute this result to that the design of RepVGG introduces useful prior knowledge. Inspired by this, they introduce a learnable gating unit in RepVGG to control the amount of information from previous layer. This design is mainly for tiny objects or dense objects to adaptively fuse the features with different receptive fields, which can be formulated as follows:

$$ y = f(x) + α_1 · g(x) + α_2 · x $$

where $α_1$ and $α_2$ are learnable parameters. In the RepResBlock, the shortcut path is not used so PP-YOLOER introduces only one parameter for each RepResBlock. During inference, the learnable parameter can be re-parameterized along with convolution layers so that neither the speed nor the amount of parameters changes.

## ProbIoU Loss

By modeling rotated bounding boxes as Gaussian bounding boxes, the _Bhattacharyya Coefficient_ of two Gaussian distributions is used to measure the similarity of two rotated bounding boxes in ProbIoU. GWD, KLD and KFIoU are also similarity measures based on Gaussian bounding boxes. To verify the effect of ProbIoU loss, authors choose KLD loss for experiment because KLD loss is scale invariant and suitable for anchor-free methods. Replacing ProbIoU loss with KLD loss causes a significant preformance drop from 78.14 mAP to 76.03 mAP, which indicates that ProbIoU loss is more suitable for PP-YOLOER design.
