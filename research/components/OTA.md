# OTA

2021 | [paper](https://arxiv.org/pdf/2103.14259) | _OTA: Optimal Transport Assignment for Object Detection_

Recent advances in label assignment in object detection mainly seek to independently define positive/negative training samples for each ground-truth (gt) object. In this paper, authors innovatively revisit the label assignment from a global perspective and propose to formulate the assigning procedure as an Optimal Transport (OT) problem – a well-studied topic in Optimization Theory. Concretely, they define the unit transportation cost between each demander (anchor) and supplier (gt) pair as the weighted summation of their classification and regression losses. After formulation, finding the best assignment solution is converted to solve the optimal transport plan at minimal transportation costs, which can be solved via Sinkhorn-Knopp Iteration. 

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/452c9566-e9b4-4fc5-a3e1-4d072729c6ba" alt="OTA" height="450"/>
</p>

## Optimal Transport

The Optimal Transport (OT) describes the following problem: supposing there are $m$ suppliers and $n$ demanders in a certain area. The $i$-th supplier holds $s_i$ units of goods while the $j$-th demander needs $d_j$ units of goods. Transporting cost for each unit of good from supplier $i$ to demander $j$ is denoted by $c_{ij}$ . The goal of OT problem is to find a transportation plan $π^\* = {π_{i,j} | i = 1, 2, ...m, j = 1, 2, ...n}$, according to which all goods from suppliers can be transported to demanders at a minimal transportation cost:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/972c8eaa-9147-4b84-aeef-3a0e80b6b8f5" alt="OTA_cost" height="300"/>
</p>

This is a linear program which can be solved in polynomial time. In our case, however, the resulting linear program is large, involving the square of feature dimensions with anchors in all scales. We thus address this issue by a fast iterative solution, named Sinkhorn-Knopp.

## OT for Label Assignment

In the context of object detection, supposing there are $m$ $gt$ targets and $n$ anchors (across all FPN levels) for an input image $I$, we view each $gt$ as a supplier who holds $k$ units of positive labels (i.e., $s_i = k, i = 1, 2, ..., m$), and each anchor as a demander who needs one unit of label (i.e., $d_j = 1, j = 1, 2, ..., n$). The cost $c^{fg}$ for transporting one unit of positive label from $gt_i$ to anchor $a_j$ is defined as the weighted summation of their _cls_ and _reg_ losses:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d6405a8a-97fa-4974-a851-33ae84021be7" alt="OTA_losses" height="100"/>
</p>

where $θ$ stands for model‘s parameters. $P_{j}^{cls}$ and $P_j^{box}$ denote predicted _cls_ score and bounding box for $a_j$. $G_i^{cls}$ and $G_i^{box}$ denote ground truth class and bounding box for $gt$ $i$. $L_{cls}$ and $L_{reg}$ stand for cross entropy loss and IoU Loss. One can also replace these two losses with Focal Loss and GIoU/SmoothL1 Loss. $α$ is the balanced coefficient. Besides positive assigning, a large set of anchors are treated as negative samples during training. As the optimal transportation involves all anchors, we introduce another supplier – background, who only provides negative labels. In a standard OT problem, the total supply must be equal to the total demand. We thus set the number of negative labels that background can supply as $n − m × k$. The cost for transporting one unit of negative label from background to $a_j$ is defined as:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f497aba9-470e-48ec-b7db-122f168e2554" alt="OTA_losses_bg" height="60"/>
</p>

where $∅$ means the background class. Concatenating this $c^{bg} ∈ R^{1 × n}$ to the last row of $c^{fg} ∈ R^{m × n}$, we can get the complete form of the cost matrix $c ∈ R^{(m+1) × n}$. The supplying vector $s$ should be correspondingly updated as:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/b4a2b139-6641-42c8-81a4-708a9105a7d9" alt="OTA_supplie_vector" height="60"/>
</p>

As we already have the cost matrix $c$, supplying vector $s ∈ R^{m+1}$ and demanding vector $d ∈ R^n$, the optimal transportation plan $π^\* ∈ R^{(m+1) × n}$ can be obtained by solving this OT problem via the off-the-shelf Sinkhorn-Knopp Iteration. After getting $π^\*$, one can decode the corresponding label assigning solution by assigning each anchor to the supplier who transports the largest amount of labels to them. The subsequent processes (e.g., calculating losses based on assigning result, back-propagation) are exactly the same as in FCOS and ATSS. Noted that the optimization process of OT problem only contains some matrix multiplications which can be accelerated by GPU devices, hence OTA only increases the total training time by less than 20% and is totally cost-free in testing phase

## Designs

**Center Prior** - Previous works only select positive anchors from the center region of objects with limited areas, called Center Prior. This is because they suffer from either a large number of ambiguous anchors or poor statistics in the subsequent process. Instead of relying on statistical characteristics, the OTA is based on global optimization methodology and thus is naturally resistant to these two issues. Theoretically, OTA can assign any anchor within the region of gts’ boxes as a positive sample. However, for general detection datasets like COCO, authors find the Center Prior still benefit the training of OTA. Forcing detectors focus on potential positive areas (i.e., center areas) can help stabilize the training process, especially in the early stage of training, which will lead to a better final performance. Hence, authors impose a Center Prior to the cost matrix. For each $gt$, they select $r^2$ closest anchors from each FPN level according to the center distance between anchors and $gt$s'. As for anchors not in the $r^2$ closest list, their corresponding entries in the cost matrix $c$ will be subject to an additional constant cost to reduce the possibility they are assigned as positive samples during the training stage.

**Dynamic k Estimation** - Intuitively, the appropriate number of positive anchors for each $gt$ (i.e., $s_i$) should be different and based on many factors like objects’ sizes, scales, and occlusion conditions, etc. As it is hard to directly model a mapping function from these factors to the positive anchor’s number, authors propose a simple but effective method to roughly estimate the appropriate number of positive anchors for each $gt$ based on the IoU values between predicted bounding boxes and $gts$. Specifically, for each $gt$, they select the top $q$ predictions according to IoU values. These IoU values are summed up to represent this $gts$ estimated number of positive anchors. Authors name this method as Dynamic k Estimation. Such an estimation method is based on the following intuition: The appropriate number of positive anchors for a certain $gt$ should be positively correlated with the number of anchors that well-regress this $gt$.