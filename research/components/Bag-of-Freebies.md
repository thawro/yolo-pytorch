# Bag Of Freebies

2019 | [paper](https://arxiv.org/pdf/1902.04103) | _Bag of Freebies for Training Object Detection Neural Networks_

Authors explore training tweaks that apply to various models including Faster R-CNN and YOLOv3. These tweaks do not change the model architectures, therefore, the inference costs remain the same.

## Bag of Freebies

* **Visually Coherent Image Mixup for Object Detection** - the distribution of blending ratio in original _mixup_ algorithm is drawn from a beta distribution $B(0.2, 0.2)$. The majority of mixups are barely noises with such beta distributions. Authors choose a beta distribution with $α$ and $β$ that are both at least 1, which is more visually coherent, instead of following the same practice in image classification. In particular, they use geometry preserved alignment for image mixup to avoid distort images at the initial steps. Beta distribution with $α$ and $β$ both equal to 1.5 is marginally better than 1.0 (equivalent to uniform distribution) and better than fixed even mixup.

* **Classification Head Label Smoothing** - modify the classifica tion loss by comparing the output distribution $p$ against the
ground truth distribution $q$ with cross-entropy $L = − \sum_{i} q_i log(p_i) $. $q$ is often a one-hot distribution, where the correct class has probability one while all other classes have zero. Soft-max function, however, can only approach this distribution when $z_i >> z_j , ∀j \neq i$ but never reach it. This encourages the model to be too confident in its predictions and is prone to over-fitting. Label smoothing was proposed as a form of regularization. The ground truth distribution is smoothed with

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/709e07b6-2904-4566-8b4a-44414da4404c" alt="bof_label_smoothing" height="100"/>
</p>

where $K$ is the total number of classes and $ε$ is a small constant. This technique reduces the model’s confidence, measured by the difference between the largest and smallest logits. 

* **Data Preprocessing** - authors check various data processing techniques, including:
  * Random geometry transformation - random cropping (with constraints), random expansion, random horizontal flip and random resize (with random interpolation).
  * Random color jittering - including brightness, hue, saturation, and contrast.
* **Training Schedule Revamping** - training with cosine schedule and proper warmup
* **Synchronized Batch Normalization**
* **Random shapes training** - a mini-batch of $N$ training images is resized to $N × 3 × H × W$ , where $H$ and $W$ are multipliers of network stride. For example, $H = W ∈ {320, 352, 384, 416, 448, 480, 512, 544, 576, 608}$ for YOLOv3 training given the stride of feature map is 32.
