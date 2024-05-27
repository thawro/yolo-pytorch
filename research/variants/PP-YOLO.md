# PP-YOLO

2020 | [paper](https://arxiv.org/pdf/2007.12099) | _PP-YOLO: An Effective and Efficient Implementation of Object Detector_

The goal of this paper is to implement an object detector with relatively balanced effectiveness and efficiency that can be directly applied in actual application scenarios, rather than propose a novel detection model. Considering that YOLOv3 has been widely used in practice, authors develop a new object detector based on YOLOv3. They mainly try to combine various existing tricks that almost not increase the number of model parameters and FLOPs, to achieve the goal of improving the accuracy of detector as much as possible while ensuring that the speed is almost unchanged.

An one-stage anchor-based detector is normally made up of a backbone network, a detection neck, which is typically a feature pyramid network (FPN), and a detection head for object classification and localization. Those are also common components in most of the one-stage anchor-free detectors based on anchor-point. Authors first revise the detail structure of YOLOv3 and introduce a modified version which replace the backbone to ResNet50-vd-dcn, which is used as the basic baseline in this paper. Then a bunch of tricks is introduced which can improve the performance of YOLOv3 almost without losing efficiency.

## Architecture

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/05eb2b99-e48f-4f03-a7ee-d52a0ee201ba" alt="PP_YOLO" height="450"/>
</p>

**<span style="color:purple">triangle</span>** - DropBlock\
**<span style="color:red">star</span>** - SPP\
**<span style="color:orange">diamond</span>** - CoordConv

**Backbone** - the overall architecture of YOLOv3 is shown in figure above. In original YOLOv3, DarkNet-53 is first applied to extract feature maps at different scales. Since ResNet has been widely used and and has been studied more extensively, there are more different variants for selection, and it has also been better optimized by deep learning frameworks. Authors decided to replace the original backbone DarkNet-53 with ResNet50-vd in PP-YOLO. Considering, that directly replacing DarkNet-53 with ResNet50-vd would hurt the performance of YOLOv3 detector, they replaced some convolutional layers in ResNet50-vd with deformable convolutional layers. The effectiveness of Deformable Convolutional Networks (DCN) has been verified in many detection models. DCN itself will not significantly increase the number of parameters and FLOPs in the model, but in practical application, too many DCN layers will greatly increase inference time. Therefore, in order to balance the efficiency and effectiveness, only the $3 × 3$ convolution layers in the last stage were replaced with DCNs. This modified backbone is denoted as ResNet50-vd-dcn, and the output of stage 3, 4 and 5 as $C_3$, $C_4$, $C_5$.

**Detection Neck** - The FPN is used to build a feature pyramid with lateral connections between feature maps. Feature maps $C_3$, $C_4$, $C_5$ are input to the FPN module. Authors denote the output feature maps of pyramid level $l$ as $P_l$, where $l = 3, 4, 5$ in the experiments. The resolution of $P_l$ is $\frac{W}{2^l} \times \frac{H}{2^l}$ for an input image of size $W × H$. The detailed structure of FPN is shown in figure above.

**Detection Head** - the detection head of YOLOv3 is very simple. It consists of two convolutional layers. A $3 × 3$ convolutional followed by an $1 × 1$ convolutional layer is adopt to get the final predictions. The output channel of each final prediction is $3(K + 5)$, where $K$ is number of classes. Each position on each final prediction map has been associate with three different anchors. For each anchor, the first K channels are the prediction of probability for $K$ classes. The following 4 channels are the prediction for bounding box localization. The last channel is the prediction of objectness score. For classification and localization, _Cross Entropy Loss_ and $L_1$ _Loss_ is adopted correspondingly. An _Objectness Loss_ is applied to supervise objectness score, which is used to identify whether is there an object or not.

### Selection of Tricks

The tricks used in the paper are all already existing (coming from other works). This paper does not propose a novel detection method, but just focuses on combining the existing tricks to implement an effective and efficient detector. Because many tricks cannot be applied to YOLOv3 directly, authors needed to adjust them according to the its structure:

* **Larger Batch Size** - using a larger batch size can improve the stability of training and get better results. Authors changed the training batch size from 64 to 192, and adjust the training schedule and learning rate accordingly

* **EMA** - when training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values. The Exponential Moving Average (EMA) compute the moving averages of trained parameters using exponential decay. For each parameter $W$ , a shadow parameter is maintained:

$$W_{EMA} = λW_{EMA} + (1 − λ)W$$

where $λ$ is the decay. Authors apply EMA with decay $λ$ of $0.9998$ and use the shadow parameter $W_{EMA}$ for evaluation.

* **DropBlock** - a form of structured dropout where units in a contiguous region of a feature map are dropped together. Different from the original paper, authors only apply DropBlock to the FPN, since they find that adding DropBlock to the backbone will lead to a decrease of the performance. The detailed inject points of the DropBlock are marked by ”triangles” in figure above.

* **IoU Loss** - bounding box regression is the crucial step in object detection. In YOLOv3, $L_1$ loss is adopted for bounding box regression. It is not tailored to the mAP evaluation metric, which is strongly rely on Intersection over Union (IoU). IoU loss and other variations such as CIoU loss and GIoU loss have been proposed to address this problem. Different from YOLOv4, authors do not replace the $L_1$-loss with IoU loss directly, but add another branch to calculate IoU loss. They find that the improvements of various IoU loss are similar, so they choose the most basic IoU loss.

* **IoU Aware** - in YOLOv3, the classification probability and objectness score is multiplied as the final detection confidence, which do not consider the localization accuracy. To solve this problem, an IoU prediction branch is added to measure the accuracy of localization. During training, IoU aware loss is adopted to training the IoU prediction branch. During inference, the predicted IoU is multiplied by the classification probability and objectiveness score to compute the final detection confidence, which is more correlated with the localization accuracy. The final detection confidence is then used as the input of the subsequent NMS.

* **Grid Sensitive** - an effective trick introduced by YOLOv4. When decoding the coordinate of the bounding box center $x$ and $y$, in original YOLOv3, one can get them by

$$x = s · (g_x + σ(p_x))$$

$$y = s · (g_y + σ(p_y ))$$

where $σ$ is the sigmoid function, $g_x$ and $g_y$ are integers and $s$ is a scale factor. Obviously, $x$ and $y$ cannot be exactly equal to $s · g_x$ or $s · (g_x + 1)$. This makes it difficult to predict the centres of bounding boxes that just located on the grid boundary. It is possible to address this problem, by changing the equations to:

$$x = s · (g_x + α · σ(p_x) − (α − 1)/2)$$

$$y = s · (g_y + α · σ(p_y ) − (α − 1)/2)$$

where $α$ is set to 1.05 in this paper. This makes it easier for the model to predict bounding box center exactly located on the grid boundary.

* **Matrix NMS** - it is motivated by Soft-NMS, which decays the other detection scores as amonotonic decreasing function of their overlaps. However, such process is sequential like traditional Greedy NMS and could not be implemented in parallel. Matrix NMS views this process from another perspective and implement it in a parallel manner. Therefore, the Matrix NMS is faster than traditional NMS, which will not bring any loss of efficiency

* **CoordConv** - it works by giving convolution access to its own input coordinates through the use of extra coordinate channels. CoordConv allows networks to learn either complete translation invariance or varying degrees of translation dependence. Considering that CoordConv will add two inputs channels to the convolution layer, some parameters and FLOPs will be added. In order to reduce the loss of efficiency as much as possible, authors do not change convolutional layers in backbone, and only replace the $1 x 1$ convolution layer in FPN and the first convolution layer in detection head with CoordConv. The detailed inject points of the CoordConv are marked by ”diamonds” in figure above.

* **SPP** - Spatial Pyramid Pooling (SPP) integrates spatial pyramid matching into CNN and use _max-pooling_ operation instead of _bag-of-words_ operation. YOLOv4 apply SPP module by concatenating _max-pooling_ outputs with kernel size $k × k$, where $k = {1, 5, 9, 13}$, and $stride = 1$. Under this design, a relatively large $k × k$ _max-pooling_ effectively increase the receptive field of backbone feature. In detail, the SPP is only applied on the top feature map as shown in figure above with ”star” mark. No parameter are introduced by SPP itself, but the number of input channel of the following convolutional layer will increase.

* **Better Pretrained Model** - using a pretrained model with higher classification accuracy on ImageNet may result in better detection performance. Authors use the distilled ResNet50-vd model
