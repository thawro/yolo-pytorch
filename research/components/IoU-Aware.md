# IoU-Aware

2020 | [paper](https://arxiv.org/pdf/1912.05992) | _IoU-aware Single-stage Object Detector for Accurate Localization_

Single-stage object detectors have been widely applied in many computer vision applications due to their simpleness and high efficiency. However, the low correlation between the classification score and localization accuracy in detection results severely hurts the average precision of the detection model. To solve this problem, an IoU-aware single-stage object detector is proposed in this paper. Specifically, IoU-aware single-stage object detector predicts the IoU for each detected box. Then the predicted IoU is multiplied by the classification score to compute the final detection confidence, which is more correlated with the localization accuracy. The detection confidence is then used as the input of the subsequent NMS and COCO AP computation, which substantially improves the localization accuracy of model

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/582abcd5-f495-43a1-a388-b0d31fb44291" alt="iou_aware" height="350"/>
</p>

The IoU-aware single-stage object detector is mostly modified from RetinaNet with the same backbone and feature pyramid network (FPN) as above figure shows. Different from the RetinaNet, an IoU prediction head parallel with the regression head is designed in the last layer of regression branch to predict the IoU for each detected box while the classification branch is kept the same. To keep the model’s efficiency, the IoU prediction head consists of only a single $3 x 3$ convolution layer, followed by a sigmoid activation layer to ensure that the predicted IoU is in the range of $[0, 1]$. There are many other choices about the design of the IoU prediction head, such as designing an independent IoU prediction branch being the same as the classification branch and regression branch, but this kind of design will severely hurt the model’s efficiency. This design brings negligible computation burden to the whole model and can still substantially improve the model’s performance

## Training

As the same as RetinaNet, the focal loss is adopted for the classification loss and the smooth $L_1$-loss is adopted for the regression loss. The binary cross-entropy loss(BCE) is adopted for the IoU prediction loss and only the losses for the positive examples are computed. $IoU_i$ represents the predicted IoU for each detected box and $\hat{IoU}_i$ is the target IoU computed between the regressed positive example $b_i$ and the corresponding ground truth box $\hat{b}_i$ . During training, whether to compute the gradient of $L_{IoU}$ with respect to $\hat{IoU}_i$ makes difference to the model’s performance. This is caused by that the gradient from IoU prediction head can be back-propagated to the regression head if the gradient of $L_{IoU}$ with respect to $\hat{IoU}_i$ is computed during training. The gradient is computed as shown in equations below. Two observations about the gradient can be obtained. Firstly, because the predicted IoU for most of the positive examples is not smaller than 0.5, the gradient is mostly non-positive and will guide the regression head to predict box $b_i$ that increases the target IoU ($\hat{IoU}_i$) which is computed between the predicted box $b_i$ and the corresponding ground truth box $\hat{b}_i$ . Secondly, as the predicted IoU ($IoU_i$) increases, the magnitude of gradient that increases the target IoU ($\hat{IoU}_i$) increases. This reduces the gap between the predicted IoU ($IoU_i$) and the target IoU ($\hat{IoU}_i$) and makes the predicted IoU more correlated with the target IoU. These two effects make our method more powerful for accurate localization as demonstrated in the following experiments. Other kinds of loss functions can also be considered, such as $L_2$-loss and $L_1$-loss. These different loss functions are compared in the following experiments. During training, the IoU prediction head is trained jointly with the classification head and regression head.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3d288711-99ec-4249-9500-4a382fcc0ef5" alt="iou_aware_loss" height="130"/>
</p>
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/81b76a4c-0358-44b2-a1a3-26de39549056" alt="iou_aware_loss_2" height="220"/>
</p>

## Inference

During inference, the classification score $p_i$ is multiplied by the predicted IoU $IoU_i$ for each detected box to calculate the final detection confidence $S_{det}$. The parameter $α$ in the range of $[0, 1]$ is designed to control the contribution of the classification score and predicted IoU to the final detection confidence. This detection confidence can simultaneously be aware of the classification score and localization accuracy and thus is more correlated with the localization accuracy than the classification score only. And it is used to rank all the detections in the subsequent NMS and AP computation. The rankings of poorly localized detections with high classification score decrease while the rankings of well localized detections with low classification score increase, thus improving the localization accuracy of the models.

$$ S_{det} = p_{i}^{\alpha} IoU_{i}^{1-\alpha} $$
