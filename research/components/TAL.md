# Task Aligned Learning

2021 | [paper](https://arxiv.org/pdf/2108.07755) | _TOOD: Task-aligned One-stage Object Detection_

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, authors propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, they design a novel Task-aligned Head (T-Head) which offers a better balance between learning task-interactive and task-specific features, as well as a greater flexibility to learn the alignment via a task-aligned predictor. Second, they propose Task Alignment Learning (TAL) to explicitly pull closer (or even unify) the optimal anchors for the two tasks during training via a designed sample assignment scheme and a task-aligned loss.

Object detection aims to identify and localize objects in images, combining the tasks of classification and localization. Traditional approaches often use separate branches for these tasks, leading to misalignment issues between classification and localization outputs. Classification focuses on learning features that identify the object, while localization aims to precisely determine the object's boundaries. The divergence in these learning mechanisms can result in spatial misalignment of features, causing inconsistencies in predictions.

Recent one-stage object detectors have attempted to address these issues by predicting consistent outputs for both tasks, often focusing on the object's center. Methods like FCOS and ATSS use a centerness branch to enhance classification scores for anchors near the object's center and assign greater weight to the localization loss for these anchors. FoveaBox, similarly, considers anchors within a central region as positive samples. These heuristic designs have improved results but still face significant challenges:

* **Independence of Classification and Localization**: One-stage detectors typically perform classification and localization independently using two parallel branches. This separation can lead to a lack of interaction between the tasks, resulting in inconsistencies. For instance, an ATSS detector might classify an anchor as a dining table while localizing a pizza more accurately, indicating a misalignment between the tasks.

* **Task-agnostic Sample Assignment**: Most anchor-free detectors use geometric assignment schemes to select anchor points near the object's center for both tasks, while anchor-based detectors assign anchor boxes based on Intersection over Union (IoU) with ground truth. These approaches often fail to align optimal anchors for classification and localization, which can vary depending on the object's shape and characteristics. This misalignment can suppress precise bounding boxes during Non-Maximum Suppression (NMS), reducing overall accuracy.

To address these limitations, we propose Task-aligned One-stage Object Detection (TOOD), which features two key innovations:

* **Task-aligned head (T-head)**: Unlike conventional heads that implement classification and localization in separate branches, our T-head enhances interaction between the tasks. By computing task-interactive features and making predictions through a Task-Aligned Predictor (TAP), T-head ensures more accurate alignment of predictions.

* **Task Alignment Learning (TAL)**: This approach explicitly pulls closer the optimal anchors for both tasks. TAL involves a sample assignment scheme that evaluates the degree of task-alignment at each anchor and a task-aligned loss that unifies the best anchors for classification and localization during training. This results in a bounding box with the highest classification score and precise localization being preserved during inference.

## Task-aligned One-stage Object Detection

**Overview** - Similar to recent one-stage detectors, the proposed TOOD has an overall pipeline of ‘backbone-FPN-head’. Moreover, by considering efficiency and simplicity, TOOD uses a single anchor per location (same as ATSS), where the ‘anchor’ means an anchor point for an anchor-free detector, or an anchor box for an anchor-based detector. As discussed, existing one-stage detectors have limitations of task misalignment between classification and localization, due to the divergence of two tasks which are often implemented using two separate head branches. In this work, authors propose to align the two tasks more explicitly using a designed **Task-aligned head (T-head)** with a new **Task Alignment Learning (TAL)**. As illustrated in figure below, **T-head** and **TAL** can work collaboratively to improve the alignment of two tasks. Specifically, T-head first makes predictions for the classification and localization on the FPN features. Then TAL computes task alignment signals based on a new task alignment metric which measures the degree of alignment between the two predictions. Lastly, T-head automatically adjusts its classification probabilities and localization predictions using learning signals computed from TAL during back propagation.

TAL_overview

### Task-aligned Head

TAL_comparison

The goal is to design an efficient head structure to improve the conventional design of the head in one-stage detectors (as shown in figure above (a)). In this work, authors achieved this by considering two aspects: 

* increasing the interaction between the two tasks
* enhancing the detector ability of learning the alignment

The proposed T-head is shown in figure above (b), where it has a simple feature extractor with two Task-Aligned Predictors (TAP). To enhance the interaction between classification and localization, authors use a feature extractor to learn a stack of task-interactive features from multiple convolutional layers, as shown by the blue part in figure above (b). This design not only facilitates the task interaction, but also provides multi-level features with multi-scale effective receptive fields for the two tasks. Formally, let $X^{fpn} ∈ R^{H × W × C}$ denotes the FPN features, where $H$, $W$ and $C$ indicate height, width and the number of channels, respectively. The feature extractor uses $N$ consecutive conv layers with activation functions to compute the task-interactive features.

TAL_features

where $conv_k$ and $δ$ refer to the $k$-th $conv$ layer and a ReLU function, respectively. Thus TAL extracts rich multi-scale features from the FPN features using a single branch in the head. Then, the computed task-interactive features will be fed into two TAP for aligning classification and localization.

#### Task-aligned Predictor (TAP)

TAL performs both object classification and localization on the computed task-interactive features, where the two tasks can well perceive the state of each other. However, due to the single branch design, the task-interactive features inevitably introduce a certain level of feature conflicts between two different tasks. Intuitively, the tasks of object classification and localization have different targets, and thus focus on different types of features (e.g., different levels or receptive fields). Consequently, authors propose a layer attention mechanism to encourage task decomposition by dynamically computing such task-specific features at the layer level. As shown in figure above (c), the task-specific features are computed separately for each task of classification or localization:

$$ X^{task}_k = w_k * X^{inter}_k, ∀k ∈ {1, 2, ..., N } $$

where $w_k$ is the $k$-th element of the learned layer attention $w ∈ R^N$. $w$ is computed from the cross-layer task-interactive features, and is able to capture the dependencies between layers:

$$ w = σ(fc_2(δ(fc_1(x^{inter})))) $$

where $fc_1$ and $fc_2$ refer to two fully-connected layers. $σ$ is a sigmoid function, and $x^{inter}$ is obtained by applying an _average pooling_ to $X^{inter}$ which are the concatenated features of $X^{inter}_k$. Finally, the results of classification or localization are predicted from each $X^{task}$:

$$ Z^{task} = conv_2(δ(conv_1(X^{task}))) $$

where $X^{task}$ is the concatenated features of $X^{task}_k$ , and $conv_1$ is a $1 × 1$ $conv$ layer for dimension reduction. $Z^{task}$ is then converted into dense classification scores $P ∈ R^{H × W × 80}$ using sigmoid function, or object bounding boxes $B ∈ R^{H × W × 4}$ with distance-to-bbox conversion.

#### Prediction alignment

At the prediction step, authors further align the two tasks explicitly by adjusting the spatial distributions of the two predictions: $P$ and $B$. Different from the previous works using a centerness branch (FCOS) or an IoU branch (IoU-Aware) which can only adjust the classification prediction based on either classification features or localization features, authors of TAL align the two predictions by considering both tasks jointly using the computed task-interactive features. Notably, they perform the alignment method separately on the two tasks. As shown in figure above (c), they use a spatial probability map $M ∈ R^{H × W × 1}$ to adjust the classification prediction:

$$ P^{align} = \sqrt{P × M} $$

where $M$ is computed from the interactive features, allowing it to learn a degree of consistency between the two tasks at each spatial location. Meanwhile, to make an alignment on localization prediction, authors further learn spatial offset maps $O ∈ R^{H × W × 8}$ from the interactive features, which are used to adjust the predicted bounding box at each location. Specifically, the learned spatial offset enables the most aligned anchor point to identify the best boundary predictions around it:

$$ B^{align}(i, j, c) = B(i + O(i, j, 2 × c), j + O(i, j, 2 × c + 1), c) $$

where an index $(i, j, c)$ denotes the $(i, j)$-th spatial location at the $c$-th channel in a tensor. Equation above is implemented by bilinear interpolation, and its computational overhead is negligible due to the very small channel dimension of $B$. Noteworthily, offsets are learned independently for each channel, which means each boundary of the object has its own learned offset. This allows for a more accurate prediction of the four boundaries because each of them can individually learn from the most precise anchor point near it. Therefore, the proposed method not only aligns the two tasks, but also improves the localization accuracy by identifying a precise anchor point for each side. The alignment maps $M$ and $O$ are learned automatically from the stack of interactive features

$$ M = σ(conv_2(δ(conv_1(X^{inter})))) $$

$$ O = conv_4(δ(conv_3(X^{inter}))) $$

where $conv_1$ and $conv_3$ are two $1 × 1$ $conv$ layers for dimension reduction. The learning of $M$ and $O$ is performed by using the proposed Task Alignment Learning (TAL) which will be described next. Notice that the proposed T-head is an independent module and can work well without TAL. It can be readily applied to various one-stage object detectors in a plug-and-play manner to improve detection performance.

### Task Alignment Learning

Authors further introduce a Task Alignment Learning (TAL) that further guides the T-head to make task-aligned predictions. TAL differs from previous methods in two aspects:

* from the task-alignment point of view, it dynamically selects high-quality anchors based on a designed metric
* it considers both anchor assignment and weighting simultaneously. It comprises a sample assignment strategy and new losses designed specifically for aligning the two tasks.

#### Task-aligned Sample Assignment

To cope with NMS, the anchor assignment for a training instance should satisfy the following rules:

1. a well-aligned anchor should be able to predict a high classification score with a precise localization jointly
2. a misaligned anchor should have a low classification score and be suppressed subsequently.

With the two objectives, authors design a new anchor alignment metric to explicitly measure the degree of task-alignment at the anchor level. The alignment metric is integrated into the sample assignment and loss functions to dynamically refine the predictions at each anchor.

**Anchor alignment metric**. Considering that a classification score and an IoU between the predicted bounding box and the ground truth indicate the quality of the predictions by the two tasks, the degree of task-alignment is measured using a high-order combination of the classification score and the IoU. To be specific, following metric is designed to compute anchor-level alignment for each instance:

$$ t = s^α × u^β $$

where $s$ and $u$ denote a classification score and an IoU value, respectively. $α$ and $β$ are used to control the impact of the two tasks in the anchor alignment metric. Notably, $t$ plays a critical role in the joint optimization of the two tasks towards the goal of task-alignment. It encourages the networks to dynamically focus on high-quality (i.e., task-aligned) anchors from the perspective of joint optimization.

**Training sample assignment**. training sample assignment is crucial to the training of object detectors. To improve the alignment of two tasks, authors focus on the task-aligned anchors, and adopt a simple assignment rule to select the training samples: **_for each instance, select $m$ anchors having the largest $t$ values as positive samples, while using the remaining anchors as negative ones_**. Again, the training is performed by computing new loss functions designed specifically for aligning the tasks of classification and localization.

#### Task-aligned Loss

**Classification objective**. To explicitly increase classification scores for the aligned anchors, and at the same time, reduce the scores of the misaligned ones (i.e., having a small $t$), authors use $t$ to replace the binary label of a positive anchor during training. However, they found that the network cannot converge when the labels (i.e., $t$) of the positive anchors become small with the increase of $α$ and $β$. Therefore, they use a normalized $t$, namely $\hat{t}$, to replace the binary label of the positive anchor, where $\hat{t}$ is normalized by the following two properties:

* to ensure effective learning of hard instances (which usually have a small $t$ for all corresponding positive anchors)
* to preserve the rank between instances based on the precision of the predicted bounding boxes.

Thus, authors adopt a simple instance-level normalization to adjust the scale of $\hat{t}$: the maximum of $\hat{t}$ is equal to the largest IoU value ($u$) within each instance. Then Binary Cross Entropy (BCE) computed on the positive anchors for the classification task can be rewritten as

$$ L_{cls\_pos} = \sum_{i=1}^{N_{pos}} BCE(s_i, \hat{t}_i) $$

where $i$ denotes the $i$-th anchor from the $N_{pos}$ positive anchors corresponding to one instance. Authors employ a focal loss for classification to mitigate the imbalance between the negative and positive samples during training. The focal loss computed on the positive anchors can be reformulated by previous equation, and the final loss function for the classification task is defined as follows

$$ L_{cls} = \sum_{i=1}^{N_{pos}} ∣\hat{t}_i − s_i ∣^γ BCE(s_i, \hat{t}_i) + \sum_{j=1}^{N_{neg}} s^γ_j BCE(s_j, 0) $$

where $j$ denotes the $j$-th anchor from the $N_{neg}$ negative anchors, and $γ$ is the focusing parameter.

**Localization objective**. A bounding box predicted by a well-aligned anchor (i.e., having a large $t$) usually has both a large classification score with a precise localization, and such a bounding box is more likely to be preserved during NMS. In addition, $t$ can be applied for selecting high- quality bounding boxes by weighting the loss more carefully to improve the training. Learning from high-quality bounding boxes is beneficial to the performance of a model, while the low-quality ones often have a negative impact on the training by producing a large amount of less informative and redundant signals to update the model. In TAL case, authors apply the $t$ value for measuring the quality of a bounding box. Thus, improve the task alignment and regression precision by focusing on the well-aligned anchors (with a large $t$), while reducing the impact of the misaligned anchors (with a small $t$) during bounding box regression. Similar to the classification objective, they re-weight the loss of bounding box regression computed for each anchor based on $\hat{t}_i$, and a GIoU loss ($L_{GIoU}) can be reformulated as follows:

$$ L_{reg} = \sum_{i=1}^{N_{pos}} \hat{t}_i L_{GIoU}(b_i, \overline{b_i}) $$

where $b$ and $\overline{b}$ denote the predicted bounding boxes and the corresponding ground-truth boxes. The total training loss for TAL is the sum of $L_{cls}$ and $L_{reg}$.