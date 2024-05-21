# Focal Loss

2018 | [paper](https://arxiv.org/pdf/1708.02002v2.pdf) | _Focal Loss for Dense Object Detection_

In this paper, the authors addressed the challenge of low accuracy in one-stage object detectors compared to two-stage detectors. They identify the imbalance between foreground and background classes during training as the main issue. To mitigate this, they introduce Focal Loss, which prioritizes hard examples, leading to their proposed detector, RetinaNet, achieving both the speed of traditional one-stage detectors and surpassing the accuracy of all existing two-stage detectors

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/17c08773-46e7-46c2-8ce4-724c6498e899" alt="focal_loss_fig" height="270"/>
</p>

In R-CNN-like detectors, class imbalance is addressed by a two-stage cascade and sampling heuristics. The proposal stage (e.g., Selective Search, EdgeBoxes, DeepMask, RPN) rapidly narrows down the number of candidate object locations to a small number (e.g., 1-2k). In the second classification stage, sampling heuristics, such as a fixed foreground-to-background ratio (1:3), or online hard example mining (OHEM), are performed to maintain a manageable balance between foreground and background

In contrast, a one-stage detector must process a much larger set of candidate object locations regularly sampled across an image. In practice this often amounts to enumerating ∼100k locations that densely cover spatial positions, scales, and aspect ratios. While similar sampling heuristics may also be applied, they are inefficient as the training procedure is still dominated by easily classified background examples. This inefficiency is a classic problem in object detection that is typically addressed via techniques such as bootstrapping or hard example mining.

Focal Loss proposed in this paper acts as a more effective alternative to previous approaches for dealing with class imbalance. The loss function is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases (see above). Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples. Experiments show that the proposed Focal Loss enables to train a high-accuracy, one-stage detector that significantly outperforms the alternatives of training with the sampling heuristics or hard example mining, the previous SoTA techniques for training one-stage detectors. To demonstrate the effectiveness of the proposed focal loss, authors designed a simple one-stage object detector called _RetinaNet_, named for its dense sampling of object locations in an input image. Its design features an efficient in-network feature pyramid and use of anchor boxes. It draws on a variety of recent ideas from SSD, RPN and FPN.

The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training (e.g., 1:1000).

More formally, authors added a modulating factor $(1 − p_t)^γ$ to the cross entropy loss, with tunable focusing parameter $γ ≥ 0$ and defined the Focal Loss as:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2c67cbcc-6837-4135-8e66-0bf1a28aec7a" alt="focal_loss_eq" height="60"/>
</p>

The focal loss is visualized for several values of $γ ∈ [0, 5]$ in figure above. Authors noted two properties of the focal loss:
* When an example is misclassified and _pt_ is small, the modulating factor is near 1 and the loss is unaffected. As $p_t → 1$, the factor goes to $0$ and the loss for well-classified examples is down-weighted
* The focusing parameter $γ$ smoothly adjusts the rate at which easy examples are down-weighted. When $γ = 0$, _FL_ is equivalent to _CE_, and as $γ$ is increased the effect of the modulating factor is likewise increased (found $γ = 2$ to work best in experiments). Intuitively, the modulating factor reduces the loss contribution from easy examples and extends the range in which an example receives low loss. For instance, with $γ = 2$, an example classified with $p_t = 0.9$ would have $100×$ lower loss compared with _CE_ and with $pt ≈ 0.968$ it would have $1000×$ lower loss. This in turn increases the importance of correcting misclassified examples (whose loss is scaled down by at most $4×$ for $p_t ≤ .5$ and $γ = 2$). In practice authors used an _α-balanced_ variant of the focal loss:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/372b592a-7464-40ac-96d0-1004801da292" alt="focal_loss_alpha_eq" height="60"/>
</p>

and adopted this form in the experiments as it yields slightly improved accuracy over the _non-α-balanced_ form. Finally, they note that the implementation of the loss layer combines the sigmoid operation for computing _p_ with the loss computation, resulting in greater numerical stability.

Another huge contribution of the paper is the RetinaNet architecture shown below:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/0b044e91-ea60-4801-a6be-d74f8bc999b5" alt="retina_net" height="400"/>
</p>

> **_NOTE:_** Some of the YOLO approaches check if using Focal Loss helps to achieve better results, but in most cases it does not. YOLO is already handling background examples via set of multipliers for loss calculation and/or picking bboxes for loss calculation. 
