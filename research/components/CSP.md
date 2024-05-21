# CSP

2019 | [paper](https://arxiv.org/pdf/1911.11929.pdf) | _CSPNet: A new backbone that can enhance learning capability of CNN_

In this paper, authors propose Cross Stage Partial Network (**CSPNet**) to mitigate the problem that previous works require heavy inference computations from the network architecture perspective. They attribute the problem to the duplicate gradient information within network optimization. The proposed networks respect the variability of the gradients by integrating feature maps from the beginning and the end of a network stage, which, in the experiments, reduces computations by 20% with equivalent or even superior accuracy on the ImageNet dataset, and significantly outperforms SoTA approaches in terms of AP50 on the MS COCO object detection dataset.

The main purpose of designing CSPNet is to enable this architecture to achieve a richer gradient combination while reducing the amount of computation. This aim is achieved by partitioning feature map of the base layer into two parts and then merging them through a proposed cross-stage hierarchy. The main concept is to make the gradient flow propagate through different network paths by splitting the gradient flow. In this way, authors have confirmed that the propagated gradient information can have a large correlation difference by switching concatenation and transition steps. In addition, CSPNet can greatly reduce the amount of computation, and improve inference speed as well as accuracy. The proposed CSPNet-based object detector deals with the following three problems:

* **Strengthening learning ability of a CNN** - the accuracy of existing CNN is greatly degraded after lightweightening, so authors hoped to strengthen CNN’s learning ability, so that it can maintain sufficient accuracy while being lightweighted
* **Removing computational bottlenecks** - too high computational bottleneck will result in more cycles to complete the inference process, or some arithmetic units will often idle. Therefore, authors hoped that it is possible to evenly distribute the amount of computation at each layer in CNN so that they could effectively upgrade the utilization rate of each computation unit and thus reduce unnecessary energy consumption. It is noted that the proposed CSPNet makes the computational bottlenecks of PeleeNet cut into half. Moreover, in the MS COCO dataset-based object detection experiments, the proposed model can effectively reduce 80% computational bottleneck when test on YOLOv3-based models
* **Reducing memory costs** - the wafer fabrication cost of Dynamic Random-Access Memory (DRAM) is very expensive, and it also takes up a lot of space. If one can effectively reduce the memory cost, he/she will greatly reduce the cost of ASIC. In addition, a small area wafer can be used in a variety of edge computing devices. In reducing the use of memory usage, authors adopt cross-channel pooling to compress the feature maps during the feature pyramid generating process. In this way, the proposed CSPNet with the proposed object detector can cut down 75% memory usage on PeleeNet when generating feature pyramids

## Cross Stage Partial

CSP can be applied to many different architectures, e.g. for DenseNet it looks as following:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8b2f5b4b-0e56-47d4-98f3-d7619321094b" alt="CSP_densenet" height="300"/>
</p>

**DenseNet** - Each stage of a DenseNet contains a dense block and a transition layer, and each dense block is composed of $k$ dense layers. The output of the $i$-th dense layer will be concatenated with the input of the $i$-th dense layer, and the concatenated outcome will become the input of the $(i + 1)$-th dense layer

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/b182d102-e3c4-4a73-a813-f740389a7d48" alt="CSP_dense_0" height="100"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/16606c4c-ec5d-4291-a0eb-40d34e9d0390" alt="csp_dense_1" height="100"/>
</p>

The equations above show forward pass of DenseNet and how to gradients are calculated. We can find that large amount of gradient information are reused for updating weights of different dense layers. This will result in different dense layers repeatedly learn copied gradient information.

**Cross Stage Partial DenseNet** - the architecture of one-stage of the proposed CSPDenseNet is shown in figure above. A stage of CSPDenseNet is composed of a partial dense block and a partial transition layer. In a partial dense block, the feature maps of the base layer in a stage are split into two parts through channel $x_0 = [x^{′}_0, x^{′′}_0]$. Between $x^{′′}_0$ and $x^{′}_0$, the former is directly linked to the end of the stage, and the latter will go through a dense block. All steps involved in a partial transition layer are as follows:

* First, the output of dense layers, $[x^{′′}_0 , x_1, ..., x_k]$, will undergo a transition layer
* Second, the output of this transition layer, $x_T$ , will be concatenated with $x^{′′}_0$ and undergo another transition layer, and then generate output $x_U$

The equations of feed-forward pass and weight updating of CSPDenseNet are shown below.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/054b206c-8e84-4faf-8ed7-34e33ca911a3" alt="csp_csp_dense_0" height="100"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2ae4d07b-6c3b-42f2-aaa3-b909c3367188" alt="csp_csp_dense_1" height="100"/>
</p>

We can see that the gradients coming from the dense layers are separately integrated. On the other hand, the feature map $x^{′}_0$ that did not go through the dense layers is also separately integrated. As to the gradient information for updating weights, both sides do not contain duplicate gradient information that belongs to other sides.

Overall speaking, the proposed CSPDenseNet preserves the advantages of DenseNet’s feature reuse characteristics, but at the same time prevents an excessively amount of duplicate gradient information by truncating the gradient flow. This idea is realized by designing a hierarchical feature fusion strategy and used in a partial transition layer.

The figure below shows the effect of applying CSP to DenseBlock:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/1da99aab-5b9e-48a0-b865-16120ae63edf" alt="csp_vs_dense" height="200"/>
</p>

**Partial Transition Layer** - the purpose of designing partial transition layers is to maximize the difference of gradient combination. The partial transition layer is a hierarchical feature fusion mechanism, which uses the strategy of truncating the gradient flow to prevent distinct layers from learning duplicate gradient information. By using the split and merge strategy across stages, it is possible to effectively reduce the possibility of duplication during the information integration process.

### Applying CSP to Other Architectures

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/094aef48-b87d-4040-9721-7c021a6cec24" alt="csp_resnext" height="200"/>
</p>

## Exact Fusion Model

Another contribution of this paper is the Exact Fusion Model (EFM) that captures an appropriate Field of View (FoV) for each anchor, which enhances the accuracy of the one-stage object detector. For segmentation tasks, since pixel-level labels usually do not contain global information, it is usually more preferable to consider larger patches for better information retrieval. However, for tasks like image classification and object detection, some critical information can be obscure when observed from image-level and bounding box-level labels. Earlier works found that CNN can be often distracted when it learns from image-level labels and concluded that it is one of the main reasons that two-stage object detectors outperform one-stage object detectors.

**Aggregate Feature Pyramid** the proposed EFM is able to better aggregate the initial feature pyramid. The EFM is based on YOLOv3, which assigns exactly one bounding-box prior to each ground truth object. Each ground truth bounding box corresponds to one anchor box that surpasses the threshold IoU. If the size of an anchor box is equivalent to the FoV of the grid cell, then for the grid cells of the sth scale, the corresponding bounding box will be lower bounded by the $(s − 1)$-th scale and upper bounded by the $(s + 1)$-th scale. Therefore, the EFM assembles features from the three scales.

**Balance Computation** - since the concatenated feature maps from the feature pyramid are enormous, it introduces a great amount of memory and computation cost. To alleviate the problem, authors incorporated the Maxout technique to compress the feature maps.

The figure below shows comparison of EFM with other feature pyramid fusion strategies.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/c25f0349-4963-406d-8ace-bb3a8c4a63ab" alt="CSP_EFM" height="200"/>
</p>

> **_NOTE:_** Most of the new YOLO approaches (mostly from the same authors as CSP) use **CSP** only to update the computational blocks.
