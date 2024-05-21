# Varifocal Loss

2021 | [paper](https://arxiv.org/pdf/2008.13367) | _VarifocalNet: An IoU-aware Dense Object Detector_

Accurately ranking the vast number of candidate detections is crucial for dense object detectors to achieve high performance. Prior work uses the classification score or a combination of classification and predicted localization scores to rank candidates. However, neither option results in a reliable ranking, thus degrading detection performance. In this paper, authors propose to learn an **Iou-aware Classification Score (IACS)** as a joint representation of object presence confidence and localization accuracy. They show that dense object detectors can achieve a more accurate ranking of candidate detections based on the IACS. Authors design a new loss function, named **Varifocal Loss**, to train a dense object detector to predict the IACS, and propose a new star-shaped bounding box feature representation for IACS prediction and bounding box refinement. Combining these two new components and a bounding box refinement branch, they build an IoU-aware dense object detector based on the FCOS + ATSS architecture, that is called **VarifocalNet** or VFNet for short.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/32861675-29bc-4103-b6c5-01798c99c109" alt="vfl_star" height="350"/>
</p>

Generally, the classification score is used to rank the bounding box in NMS. However, this harms the detection performance, because the classification score is not always a good estimate of the bounding box localization accuracy and accurately localized detections with low classification scores may be mistakenly removed in NMS. To solve the problem, existing dense object detectors predict either an additional IoU score (IoU-Aware) or a centerness score (FCOS) as the localization accuracy estimation, and multiply them by the classification score to rank detections in NMS. These methods can alleviate the misalignment problem between the classification score and the object localization accuracy. However, they are sub-optimal because multiplying the two imperfect predictions may lead to a worse rank basis. Besides, adding an extra network branch to predict the localization score is not an elegant solution and incurs additional computation burden.

Authors of Varifocal Loss proceed with answering the question: _Instead of predicting an additional localization accuracy score, can we merge it into the classification score?_ That is, predict a localization-aware or IoU-aware classification score (IACS) that simultaneously represents the presence of a certain object class and the localization accuracy of a generated bounding box

## VarifocalNet

Based on the discovery above, authors propose to learn the IoU-aware classification score (IACS) to rank detections. They build a new dense object detector, coined as VarifocalNet or VFNet, based on the FCOS + ATSS with the centerness branch removed. Compared with the FCOS + ATSS, it has three new components: the varifcoal loss, the star-shaped bounding box feature representation and the bounding box refinement.

### IACS – IoU-Aware Classification Score

IACS is defined as a scalar element of a classification score vector, in which the value at the ground-truth class label position is the IoU between the predicted bounding box and its ground truth, and 0 at other positions.

### Varifocal Loss (VFL)

A novel Varifocal Loss is designed for training a dense object detector to predict the IACS. Since it is inspired by Focal Loss, a brief review of FL is presented. Focal loss is designed to address the extreme imbalance problem between foreground and background classes during the training of dense object detectors. It is defined as:

$$
FL(p, y) =  
\begin{cases}
-\alpha(1 - p)^{\gamma} log(p) \space \space \space \space \space \space \space \space \space \space \textrm{if y = 1} \\
-(1 - \alpha)p^{\gamma} log(1 - p) \space \space \space \textrm{otherwise}
\end{cases}
$$

where $y ∈ {±1}$ specifies the ground-truth class and $p ∈ [0, 1]$ is the predicted probability for the foreground class. As shown in Equation above, the modulating factor ($(1 − p)^γ$ for the foreground class and $p^γ$ for the background class) can reduce the loss contribution from easy examples and relatively increases the importance of mis-classified examples. Thus, the **focal loss prevents the vast number of easy negatives from overwhelming the detector during training and focuses the detector on a sparse set of hard examples**. VFL borrows the example weighting idea from the FL to address the class imbalance problem when training a dense object detector to regress the continuous IACS. However, unlike the FL that deals with positives and negatives equally, VFL treat them **asymmetrically**. The proposed Varifocal Loss is also based on the binary cross entropy loss and is defined as:

$$
VFL(p, q) =  
\begin{cases}
-q(qlog(p) + (1-q)log(1-p)) \quad \textrm{q > 0} \\
-\alpha p^{\gamma} log(1-p) \qquad \qquad \qquad \qquad \textrm{q = 0}
\end{cases}
$$

where $p$ is the predicted IACS and $q$ is the target score. For a foreground point, $q$ for its ground-truth class is set as the IoU between the generated bounding box and its ground truth (gt\_IoU) and 0 otherwise, whereas for a background point, the target $q$ for all classes is 0. See Figure below.

vfl_star

As VFL Equation shows, the varifocal loss only reduces the loss contribution from negative examples ($q = 0$) by scaling their losses with a factor of $p^γ$ and does not down-weight positive examples ($q > 0$) in the same way. This is because positive examples are extremely rare compared with negatives and we should keep their precious learning signals. On the other hand, inspired by other works, authors weight the positive example with the training target $q$. If a positive example has a high gt\_IoU, its contribution to the loss will thus be relatively big. This focuses the training on those high-quality positive examples which are more important for achieving a higher AP than those low-quality ones. To balance the losses between positive examples and negative examples, authors add an adjustable scaling factor $\alpha$ to the negative loss term.

### Star-Shaped Box Feature Representation

A star-shaped bounding box feature representation is designed for IACS prediction. It uses the features at nine fixed sampling points (yellow circles in first figure) to represent a bounding box with the deformable convolution. This new representation can capture the geometry of a bounding box and its nearby contextual information, which is essential for encoding the misalignment between the predicted bounding box and the ground-truth one. Specifically, given a sampling location $(x, y)$ on the image plane (or a projecting point on the feature map), we first regress an initial bounding box from it with $3 x 3$ convolution. Following the FCOS, this bounding box is encoded by a $4D$ vector $(l’, t’, r’, b’)$ which means the distance from the location $(x, y)$ to the left, top, right and bottom side of the bounding box respectively. With this distance vector, we heuristically select nine sampling points at: $(x, y)$, $(x-l’, y)$, $(x, y-t’)$, $(x+r’, y)$, $(x, y+b’)$, $(x-l’, y-t’)$, $(x+l’, y-t’)$, $(x-l’, y+b’)$ and $(x+r’, y+b’)$, and then map them onto the feature map. Their relative offsets to the projecting point of $(x, y)$ serve as the offsets to the deformable convolution and then features at these nine projecting points are convolved by the deformable convolution to represent a bounding box. Since these points are manually selected without additional prediction burden, the new representation is computation efficient.

### Bounding Box Refinement

Authors further improve the object localization accuracy through a bounding box refinement step. Bounding box refinement is a common technique in object detection, however, it is not widely adopted in dense object detectors due to the lack of an efficient and discriminative object descriptor. With the new star representation, it is possible to adopt it in dense object detectors without losing efficiency. Authors model the bounding box refinement as a residual learning problem. For an initially regressed bounding box $(l’, t’, r’, b’)$, they first extract the star-shaped representation to encode it. Then, based on the representation, they learn four distance scaling factors $(∆l, ∆t, ∆r, ∆b)$ to scale the initial distance vector, so that the refined bounding box that is represented by $(l, t, r, b) = (∆l×l’, ∆t×t’, ∆r×r’, ∆b×b’)$ is closer to the ground truth.

### VarifocalNet

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/99abfbd2-c5f2-4314-9814-4222ce30f63f" alt="vfl_net" height="350"/>
</p>

Attaching the above three components to the FCOS network architecture and removing the original centerness branch, composes the VarifocalNet. Figure above illustrates the network architecture of the VFNet. The backbone and FPN network parts of the VFNet are the same as the FCOS. The difference lies in the head structure. The VFNet head consists of two subnetworks.

* the localization subnet performs bounding box regression and subsequent refinement. It takes as input the feature map from each level of the FPN and first applies three $3 x 3$ conv layers with ReLU activations. This produces a feature map with 256 channels. One branch of the localization subnet convolves the feature map again and then outputs a $4D$ distance vector $(l’, t’, r’, b’)$ per spatial location which represents the initial bounding box. Given the initial box and the feature map, the other branch applies a star-shaped deformable convolution to the nine feature sampling points and produces the distance scaling factor $(∆l, ∆t, ∆r, ∆b)$ which is multiplied by the initial distance vector to generate the refined bounding box $(l, t, r, b)$
* the other subnet aims to predict the IACS. It has the similar structure to the localization subnet (the refinement branch) except that it outputs a vector of $C$ (the class number) elements per spatial location, where each element represents jointly the object presence confidence and localization accuracy.

### Loss Function and Inference

The training of the VFNet is supervised by the loss function:

$$ Loss =
\frac{1}{N_{pos}} \sum_i \sum_c VFL(p_{c, i}, q_{c, i}) +
\frac{\lambda_0}{N_{pos}} \sum_i q_{c^\*, i} L_{bbox}(bbox_i', bbox_i^\*) +
\frac{\lambda_1}{N_{pos}} \sum_i q_{c^\*, i} L_{bbox}(bbox_i, bbox_i^\*) $$

where $p_{c,i}$ and $q_{c,i}$ denote the predicted and target IACS respectively for the class $c$ at the location $i$ on each level feature map of FPN. $L_{bbox}$ is the GIoU loss, and $bbox_i'$, $bbox_i$ and $bbox_i^\*$ represent the initial, refined and groundtruth bounding box respectively. Authors weight the $L_{bbox}$ with the training target $q_{c^\∗, i}$, which is the gt\_IoU for foreground points and 0 otherwise, following the FCOS. $λ_0$ and $λ_1$ are the balance weights for $L_{bbox}$ and are empirically set as $1.5$ and $2.0$ respectively in this paper. $N_{pos}$ is the number of foreground points and is used to normalize the total loss. As mentioned previously, authors employ the ATSS to define foreground and background points during training

**Inference** - the inference of the VFNet is straightforward. It involves simply forwarding an input image through the network and a NMS post-processing step for removing redundant detections.
