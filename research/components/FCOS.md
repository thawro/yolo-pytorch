# FCOS

2019 | [paper](https://arxiv.org/pdf/1904.01355) | _FCOS: Fully Convolutional One-Stage Object Detection_

The proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, authors also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. 

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/dabdf5f9-672a-48da-a660-6d8f66ffc401" alt="FCOS" height="350"/>
</p>

FCOS contributions:

* **Fully Convolutional One-Stage Object Detector** - reformulates object detection in a per-pixel prediction fashion
* **Multi-level Prediction with FPN for FCOS** - make use of multi-level prediction to improve the recall and resolve the ambiguity resulted from overlapped bounding boxes
* **Center-ness branch** - additional head branch, which helps suppress the low-quality detected bounding boxes

## Fully Convolutional One-Stage Object Detector

Let $F_i ∈ R^{H × W × C}$ be the feature maps at layer $i$ of a backbone CNN and $s$ be the total stride until the layer. The ground-truth bounding boxes for an input image are defined as ${B_i}$, where $B_i = (x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)}, c^{(i)}) ∈ R^{4} \times \{1, 2, ..., C\}$. Here $(x_0^{(i)}, y_0^{(i)})$  and $(x_1^{(i)}, y_1^{(i)})$ denote the coordinates of the left-top and right-bottom corners of the bounding box. $c^{(i)}$ is the class that the object in the bounding box belongs to. $C$ is the number of classes, which is 80 for MS-COCO dataset. 

For each location $(x, y)$ on the feature map $F_i$, we can map it back onto the input image as $(\lfloor \frac{s}{2} \rfloor + xs, \lfloor \frac{s}{2} \rfloor + ys)$, which is near the center of the receptive field of the location $(x, y)$. Different from anchor-based detectors, which consider the location on the input image as the center of (multiple) anchor boxes and regress the target bounding box with these anchor boxes as references, FCOS directly regress the target bounding box at the location. In other words, our detector directly views locations as training samples instead of anchor boxes in anchor-based detectors, which is the same as FCNs for semantic segmentation.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/6d572c35-15c8-4f49-abf8-47bafca2a057" alt="FCOS_distances" height="350"/>
</p>

Specifically, location $(x, y)$ is considered as a positive sample if it falls into any ground-truth box and the class label $c^\*$ of the location is the class label of the ground-truth box. Otherwise it is a negative sample and $c^* = 0$ (background class). Besides the label for classification, we also have a $4D$ real vector $t^\* = (l^\*, t^\*, r^\*, b^\*)$ being the regression targets for the location. Here $l^\*$, $t^\*$, $r^\*$ and $b^\*$ are the distances from the location to the four sides of the bounding box as shown in figure above. If a location falls into multiple bounding boxes, it is considered as an ambiguous sample. Authors simply choose the bounding box with minimal area as its regression target. With multi-level prediction, the number of ambiguous samples can be reduced significantly and thus they hardly affect the detection performance. Formally, if location $(x, y)$ is associated to a bounding box $B_i$, the training regression targets for the location can be formulated as:

$$ l^\* = x - x_0^{(i)} $$

$$ t^\* = y - y_0^{(i)} $$

$$ r^\* = x_1^{(i)} - x $$

$$ b^\* = y_1^{(i)} - y $$

It is worth noting that FCOS can leverage as many foreground samples as possible to train the regressor. It is different from anchor-based detectors, which only consider the anchor boxes with a highly enough IoU with ground-truth boxes as positive samples.

**Network Outputs** - corresponding to the training targets, the final layer of our networks predicts an $80D$ vector **$p$** of classification labels and a $4D$ vector $t = (l, t, r, b)$ bounding box coordinates. Instead of training a multi-class classifier, FCOS trains $C$ binary classifiers. FCOS adds four convolutional layers after the feature maps of the backbone networks respectively for classification and regression branches. Moreover, since the regression targets are always positive, FCOS employs $exp(x)$ to map any real number to $(0, ∞)$ on the top of the regression branch. _It is worth noting that FCOS has 9× fewer network output variables than the popular anchor-based detectors with 9 anchor boxes per location_.

**Loss Function**  The training loss function is defined as:

$$ L({p_{x, y}}, {t_{x, y}}) = \frac{1}{N_{pos}} \sum_{x, y} L_{cls}(p_{x, y}, c_{x, y}^\*) + \frac{\lambda}{N_{pos}} \sum_{x, y} 1_{c_{x, y}^\* > 0} L_{reg}(t_{x, y}, t_{x, y}^\*) $$

where $L_{cls}$ is focal loss and $L_{reg}$ is the IOU loss. $N_{pos}$ denotes the number of positive samples and $λ$ being 1 in this paper is the balance weight for $L_{reg}$. The summation is calculated over all locations on the feature maps $F_i$. $1_{c^\*_i > 0}$ is the indicator function, being $1$ if $c^\*_i > 0$ and $0$ otherwise.

**Inference** the inference of FCOS is straightforward. Given an input images, we forward it through the network and obtain the classification scores $p_{x,y}$ and the regression prediction $t_{x,y}$ for each location on the feature maps $F_i$. Authors choose the location with $p_{x,y} > 0.05$ as positive samples and invert above equation to obtain the predicted bounding boxes.

## Multi-level Prediction with FPN for FCOS

Two possible issues of the proposed FCOS can be resolved with multi-level prediction with FPN:

* The large stride (e.g., $16×$) of the final feature maps in a CNN can result in a relatively low best possible recall (BPR). For anchor based detectors, low re
call rates due to the large stride can be compensated to some extent by lowering the required IoU scores for positive anchor boxes. For FCOS, at the first glance one may think that the BPR can be much lower than anchor-based detectors because it is impossible to recall an object which no location on the final feature maps encodes due to a large stride. Even with a large stride, FCN-based FCOS is still able to produce a good BPR. Therefore, the BPR is actually not
a problem of FCOS. Moreover, with multi-level FPN prediction, the BPR can be further improved.

* Overlaps in ground-truth boxes can cause intractable ambiguity, i.e., which bounding box should a location in the overlap regress? This ambiguity results in degraded performance of FCN-based detectors. The ambiguity can be greatly resolved with multi-level prediction.

Following FPN, FCOS detects different sizes of objects on different levels of feature maps. Specifically, authors make use of five levels of feature maps defined as $\{P_3, P_4, P_5, P_6, P_7\}$. $P_3, P_4$ and $P_5$ are produced by the backbone CNNs’ feature maps $C_3, C_4$ and $C_5$ followed by a $1 × 1$ convolutional layer with the top-down connections, as shown in figure above. $P_6$ and $P_7$ are produced by applying one convolutional layer with the stride being 2 on $P_5$ and $P_6$, respectively. As a result, the feature levels $P_3, P_4, P_5, P_6$ and $P_7$ have strides $8, 16, 32, 64$ and $128$, respectively.

Unlike anchor-based detectors, which assign anchor boxes with different sizes to different feature levels, FCOS directly limits the range of bounding box regression for each level. More specifically, authors firstly compute the regression targets $l^\*, t^\*, r^\*, b^\*$ for each location on all feature levels. Next, if a location satisfies $max(l^\*, t^\*, r^\*, b^\*) > m_i$ or $max(l^\*, t^\*, r^\*, b^\*) < m_{i−1}$, it is set as a negative sample and is thus not required to regress a bounding box anymore. Here $m_i$ is the maximum distance that feature level $i$ needs to regress. In this work, $m_2, m_3, m_4, m_5, m_6$ and $m_7$ are set as $0, 64, 128, 256, 512$ and $∞$, respectively. Since objects with different sizes are assigned to different feature levels and most overlapping happens between objects with considerably different sizes. If a location, even with multi-level prediction used, is still assigned to more than one ground-truth boxes, FCOS simply choose the groundtruth box with minimal area as its target.

FCOS also shares the heads between different feature levels, not only making the detector parameter-efficient but also improving the detection performance. However, authors observe that different feature levels are required to regress different size range (e.g., the size range is $[0, 64]$ for $P_3$ and $[64, 128]$ for $P_4$), and therefore it is not reasonable to make use of identical heads for different feature levels. As a result, instead of using the standard $exp(x)$, authors make use of $exp(s_ix)$ with a trainable scalar $s_i$ to automatically adjust the base of the exponential function for feature level $P_i$, which slightly improves the detection performance.

## Center-ness for FCOS

After using multi-level prediction in FCOS, there is still a performance gap between FCOS and anchor-based detectors. Authors observed that it is due to a lot of low-quality predicted bounding boxes produced by locations far away from the center of an object. They proposed a simple yet effective strategy to suppress these low-quality detected bounding boxes without introducing any hyper-parameters. Specifically, they add a single-layer branch, in parallel with the classification branch (as shown in figure above) to predict the “center-ness” of a location. The center-ness depicts the normalized distance from the location to the center of the object that the location is responsible for. Given the regression targets $l^\*, t^\*, r^\*, b^\*$ for a location, the center-ness target is defined as:

$$ centerness^\* = \sqrt{\frac{min(l^\*, r^\*)}{max(l^\*, r^\*)} \times \frac{min(t^\*, b^\*)}{max(t^\*, b^\*)}} $$

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/08b5ec68-ce3c-434f-8e4e-ede2d1949830" alt="FCOS_centerness" height="350"/>
</p>

FCOS employs $sqrt$ here to slow down the decay of the center-ness. The center-ness ranges from 0 to 1 and is thus trained with binary cross entropy (BCE) loss. The loss is added to the loss function from previous equation. When testing, the final score (used for ranking the detected bounding boxes) is computed by multiplying the predicted center-ness with the corresponding classification score. Thus the center-ness can downweight the scores of bounding boxes far from the center of an object. As a result, with high probability, these low-quality bounding boxes might be filtered out by the final non-maximum suppression (NMS) process, improving the detection performance remarkably. An alternative of the center-ness is to make use of only the central portion of ground-truth bounding box as positive samples with the price of one extra hyper-parameter, as shown in other works. After the submission, it has been shown that the combination of both methods can achieve a much better performance.
