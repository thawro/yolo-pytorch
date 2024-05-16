| Year | Done | Approach | Link | Title |
| :--- | :---: | :--------------- | :---- | :---- |
| 2016 | ✅ | [YOLOv1](#YOLOv1) | [paper](https://arxiv.org/pdf/1506.02640.pdf) | You Only Look Once: Unified, Real-Time Object Detection |
| 2016 | ✅ | [YOLOv2](#YOLOv2), YOLO9000 | [paper](https://arxiv.org/pdf/1612.08242.pdf) | YOLO9000: Better, Faster, Stronger |
| 2018 | ✅ | [YOLOv3](#YOLOv3) | [paper](https://arxiv.org/pdf/1804.02767.pdf), [video](https://www.youtube.com/watch?v=Grir6TZbc1M), [blog](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), [code](https://github.com/ultralytics/yolov3) | YOLOv3: An Incremental Improvement |
| 2019 | ✅ | [Gaussian YOLO](#Gaussian-YOLO) | [paper](https://arxiv.org/pdf/1904.04620) | Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving |
| 2020 | ✅ | [PP-YOLO](#PP-YOLO) | [paper](https://arxiv.org/pdf/2007.12099) | PP-YOLO: An Effective and Efficient Implementation of Object Detector |
| 2020 | ✅ | [YOLOv4](#YOLOv4) | [paper](https://arxiv.org/pdf/2004.10934.pdf) , [blog](https://alexeyab84.medium.com/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe), [code_1](https://github.com/WongKinYiu/PyTorch_YOLOv4), [code_2](https://github.com/Tianxiaomo/pytorch-YOLOv4) | YOLOv4: Optimal Speed and Accuracy of Object Detection |
| 2020 | ✅ | [YOLOv5](#YOLOv5) | [code](https://github.com/ultralytics/yolov5) | YOLOv5 |
| 2021 | ✅ | [Scaled-YOLOv4](#Scaled-YOLOv4) | [paper](https://arxiv.org/pdf/2011.08036.pdf) , [blog](https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982), [code](https://github.com/WongKinYiu/ScaledYOLOv4) | Scaled-YOLOv4: Scaling Cross Stage Partial Network |
| 2021 | 🔳 | [YOLOR](#YOLOR) | [paper](https://arxiv.org/pdf/2105.04206), [code](https://github.com/WongKinYiu/yolor) | You Only Learn One Representation: Unified Network for Multiple Tasks |
| 2021 | 🔳 | [YOLOX](#YOLOX) | [paper](https://arxiv.org/pdf/2107.08430.pdf) | YOLOX: Exceeding YOLO Series in 2021 |
| 2022 | 🔳 | [PP-YOLOE](#PP-YOLOE) | [paper](http://arxiv.org/pdf/2203.16250v3) | PP-YOLOE: An evolved version of YOLO |
| 2022 | 🔳 | [RTMDet](#RTMDet) | [paper](http://arxiv.org/pdf/2212.07784v2) | RTMDet: An Empirical Study of Designing Real-Time Object Detectors |
| 2022 | 🔳 | [YOLOv7](#YOLOv7) | [paper](https://arxiv.org/pdf/2207.02696.pdf), [code](https://github.com/WongKinYiu/yolov7) | YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors |
| 2022 | 🔳 | [YOLOv6](#YOLOv6) | [paper](https://arxiv.org/pdf/2209.02976), [code](https://github.com/meituan/YOLOv6) | YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications |
| 2022 | 🔳 | [PP-YOLOE-R](#PP-YOLOE-R) | [paper](https://arxiv.org/pdf/2211.02386) | PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector |
| 2023 | 🔳 | [DAMA-YOLO](#DAMA-YOLO) | [paper](http://arxiv.org/pdf/2211.15444v4) | DAMO-YOLO : A Report on Real-Time Object Detection Design |
| 2023 | 🔳 | [YOLOv6 v3.0](#YOLOv6-v3.0) | [paper](https://arxiv.org/pdf/2301.05586.pdf) | YOLOv6 v3.0: A Full-Scale Reloading |
| 2023 | 🔳 | [Dynamic YOLOv7](#Dynamic-YOLOv7) | [paper](https://arxiv.org/pdf/2304.05552) | DynamicDet: A Unified Dynamic Architecture for Object Detection |
| 2023 | 🔳 | [YOLO Review](#YOLO-Review) | [paper](https://arxiv.org/pdf/2304.00501.pdf) | A comprehensive review of YOLO: from YOLOv1 and beyond |
| 2023 | 🔳 | [YOLOv8](#YOLOv8) | [code](https://github.com/ultralytics/ultralytics) | YOLOv8 |
| 2024 | 🔳 | [YOLOv9](#YOLOv9) | [paper](https://arxiv.org/pdf/2402.13616), [code](https://github.com/WongKinYiu/yolov9) | YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information |



# YOLO v1
2016 | [paper](https://arxiv.org/pdf/1506.02640.pdf) | _You Only Look Once: Unified, Real-Time Object Detection_

YOLOv1 is a **single-stage** object detection model. Object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/596ae49e-22d6-4a11-85da-d89a50df9a3b" alt="yolo_v1" height="250"/>
</p>

## How it works

* YOLO divides the input image into an $S × S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
* Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how
confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. The confidence score is defined as $Pr(Object) ∗ IoU(pred, gt)$. If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.
* Each bounding box consists of 5 predictions: $x, y, w, h$, and confidence. The $(x, y)$ coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IoU between the predicted box and any ground truth box.
* Each grid cell also predicts $C$ conditional class probabilities, $Pr(Class_i|Object)$. These probabilities are conditioned on the grid cell containing an object. We only predict one set of class probabilities per grid cell, regardless of the number of boxes $B$.
* predictions are encoded as an $S × S × (B ∗ 5 + C)$ tensor ($S$ - grid size, $B$ - number of bboxes, $C$ - number of classes). In paper: $7 × 7 × 30$, that is $S = 7, B = 2, C = 20$ (PASCAL VOC has 20 labels)

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2304ce9a-56e2-450e-be12-6eb5294b561d" alt="yolo_how" width="500"/>
</p>

## Model architecture

* The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates.
* The network architecture is inspired by the GoogLeNet. It has 24 convolutional layers followed by 2 fully connected layers
* Instead of the inception modules used by GoogLeNet, the $1 × 1$ conv reduction layers followed by $3 × 3$ convolutional layers are used
* To avoid overfitting use dropout and extensive data augmentation. A dropout layer with $rate = 0.5$ after the first connected layer prevents co-adaptation between layers

## Loss function
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/a29f820b-b37f-472e-b5ea-52456b4383eb" alt="yolo_loss" width="500"/>
</p>

## Training details

* pretrain first 20 conv layers (+ global pool and fc) as classifier on the ImageNet -> val top-5 88% accuracy
* add 4 conv layers and 2 fc layers on top of the pretrained backbone and increase input size from 224 to 448 to train the detector
* normalize bbox width and height to _[0, 1]_ and parametrize bbox x and y to be offsets of a particular grid cell location so they are also bounded in _[0, 1]_
* LeakyReLU(0.1)
* use different loss weights for case of no object cell and for coordinate losses
* use squared root of width and height in loss (to reflect that small deviations in large boxes matter less than in small boxes)
* YOLO predicts multiple bounding boxes per grid cell. At training time only one bbox predictor
is responsible for each object. One predictor is assigned to be “responsible” for predicting an object based on which prediction has the highest current IoU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall
* The loss function only penalizes classification error if an object is present in that grid cell. It also only penalizes bbox coordinate error if that predictor is “responsible” for the ground truth box (i.e. has the highest IoU of any predictor in that grid cell)
* Training scheme:
	* 135 epochs on the training and validation data sets from PASCAL VOC 2007 and 2012 (when testing on 2012 we also include the VOC 2007 test data for training)
	* batch size = 64
	* momentum = 0.9 
	* weight decay = 0.0005
	* Learning rate schedule:
		* For the first epochs slowly raise the learning rate from 0.001 to 0.01 (If started at a high learning rate the model often diverges due to unstable gradients)
		* Continue training with 0.01 for 75 epochs
		* Then 0.001 for 30 epochs
		* Finally 0.0001 for 30 epochs
	* data augmentation:
		* random scaling,
		* translations of up to 20% of the original image size
		* randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space

# YOLO v2
2016 | [paper](https://arxiv.org/pdf/1612.08242.pdf) | _YOLO9000: Better, Faster, Stronger_

YOLOv2 (or YOLO9000) is a single-stage real-time object detection model. It improves upon [_YOLOv1_](#yolo-v1) in several ways, including the use of Darknet-19 as a backbone, batch normalization, use of a high-resolution classifier, multi-scale training, the use of dimension clusters, fine-grained features and direct location prediction to predict bounding boxes, and more

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3956c34f-f205-4366-9cbb-83f2308689d6" alt="yolo_v2" height="250"/>
</p>

## How it works

It works similar to the YOLOv1, the main differences include each predicted bbox has its own C class probabilities, so the predictions are encoded as an _S × S × (B ∗ (5 + C))_ tensor and added passtrough layer, dimension clusters prior and direct location prediction for easier training.
Main improvements:
* **Batch Normalization** - leads to significant improvements in convergence while eliminating the need for other forms of regularization (removed dropout from fc). By adding batch normalization on all of the convolutional layers in YOLO mAP improved by more than 2%
* **High Resolution Classifier** - the original YOLO trains the classifier network at _224 × 224_ and increases the resolution to 448 for detection. This means the network has to simultaneously switch to learning object detection and adjust to the new input resolution. For YOLOv2 we first fine tune the classification network at the full _448 × 448_ resolution for 10 epochs on ImageNet. This gives the network time to adjust its filters to work better
on higher resolution input. Then finetune the resulting network on detection. This high resolution classification network gives an increase of almost 4% mAP
* **Dimension Clusters** - authors encountered two issues with anchor boxes when using them with YOLO. The first is that the box dimensions are hand picked. If better priors for anchor boxes are picked for the network to start with it can make it easier to learn to predict good detections. Instead of choosing priors by hand, _k-means_ clustering is used on the training set bounding boxes to automatically find good priors. Distance metric is based on IoU, that is: _d(box, centroid) = 1 − IoU(box, centroid)_. By analysis, the _k = 5_ value is chosen for the number of priors.
* **Direct location prediction** - Instead of predicting offsets (for anchors) the similar to YOLOv1 approach is used, that is predict location coordinates relative to the location of the grid cell. This bounds the ground truth to fall in _[0, 1]_. The Logistic activation function is used to constrain the network’s predictions to fall in this range. The network predicts 5 bboxes at each cell in the output feature map. The network predicts 5 coordinates for each bbox, _tx_, _ty_ , _tw_, _th_, and to. If the cell is offset from the top left corner of the image by _(cx, cy )_ and the bbox prior has width and height _pw_, _ph_, then the predictions correspond to:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/848f9316-b3a4-49db-b8bc-e6be7acbeeeb" alt="yolov2_bbox_pic" height="200"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/ca70fb43-1cfd-4b93-baf3-4c64b0c707cf" alt="yolov2_bbox_def" height="200"/>
</p>

* **Fine-Grained Features** - YOLOv2 predicts detections on a _13 × 13_ feature map. While this is sufficient for large objects, it may benefit from finer grained features for localizing smaller objects. The passthrough layer is added that brings features from an earlier layer at _26 × 26_ resolution. The passthrough layer concatenates the higher resolution features with the low resolution features by stacking adjacent features into different channels instead of spatial locations, similar to the identity mappings in ResNet. This turns the _26 × 26 × 512_ feature map into a _13 × 13 × 2048_ feature map, which can be concatenated with the original features
* **Multi-Scale Training**. The original YOLO uses an input resolution of _448 × 448_. With the addition of anchor boxes the resolution is changed to _416 × 416_. However, since the model only uses convolutional and pooling layers it can be resized on the fly. The YOLOv2 is designed to be robust to running on images of different sizes. Instead of fixing the input image size, it is changed every few iterations. Every 10 batches a new image dimension size is chosen from multiples of 32 (model downsamples by a factor of 32): {320, 352, ..., 608}  Thus the smallest option is _320 × 320_ and the largest is _608 × 608_.

## Model architecture

**Darknet-19** - a new classification model to be used as the backbone of YOLOv2. Similar to the VGG models, Darknet-19 use mostly _3 × 3_ filters and double the number of channels after every pooling step. Following the work on Network in Network (NIN) the global average pooling to is used make predictions as well as _1 × 1_ filters to compress the feature representation between _3 × 3_ convolutions. Batch normalization is used to stabilize training, speed up convergence, and regularize the model. The final model, called Darknet-19, has 19 convolutional layers and 5 maxpooling It achieves 72.9% top-1 accuracy and 91.2% top-5 accuracy on ImageNet.

## Training details:
**Training for classification** - the network is trained on the standard ImageNet 1000 class classification dataset for 160 epochs using stochastic gradient descent (SGD) with a starting learning rate of 0.1, polynomial rate decay with a power of 4, weight decay of 0.0005 and momentum of 0.9. During training the standard data augmentation tricks are used, including random crops, rotations, and hue, saturation, and exposure shifts. As discussed above, after initial training on images at _224 × 224_ the model is finetuned at a larger size, _448_ for 10 epochs starting at learning rate of 0.001. After finetuning the model achieves a top-1 accuracy of 76.5% and a top-5 accuracy of 93.3%.

**Training for detection** - for detection the last convolutional layer is removed and instead three _3 × 3_ convolutional layers are added (each with 1024 filters) and a final _1 × 1_ convolutional layer with the number of outputs needed for detection (for VOC 5 boxes are predicted, each with 5 coordinated and 20 classes, so 125 filters in total). The passtrough layer is also added from the final _3 × 3 × 512_ layer to the second to last convolutional layer so that the model can use fine grain features. The network is trained for 160 epochs with a starting
learning rate of 0.001, dividing it by 10 at 60 and 90 epochs. The weight decay is 0.0005, momentum is 0.9 and data augmentation includes random crops, color shifting, and other similar to YOLO and SSD



# YOLO v3
2018 | [paper](https://arxiv.org/pdf/1804.02767.pdf) | _YOLOv3: An Incremental Improvement_
_YOLOv3_ is a real-time, single-stage object detection model that builds on [_YOLOv2_](#yolo-v2) with several improvements. Improvements include the use of a new backbone network, Darknet-53 that utilises residual connections, as well as some improvements to the bbox prediction step, and use of three different scales from which to extract features. Ultralytics implementation can be found [here](https://github.com/ultralytics/yolov3).

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/c86504c6-7712-4ceb-a358-279ae53ebb0e" alt="yolo_v3" height="400"/>
</p>

## How it works

* Similar to YOLOv2, the model outputs a tensor of size _S × S × [B ∗ (5 + C)]_, but now the model outputs bboxes at three different scales
* Different to previous versions, YOLOv3 uses multiple independent logistic classifiers rather than one softmax layer for each class. During training, they use binary cross-entropy loss in a one vs all setup (using a softmax imposes the assumption that each box has exactly one class which is often not the case - the multilabel approach better models the data)
* The bigger backbone is used (DarkNet-53) for feature extraction - the architecture has alternative _1 × 1_ and _3 × 3_ convolution layers and skip/residual connections inspired by the ResNet model. Although DarkNet53 is smaller than ResNet101 or RestNet-152, it is faster and has equivalent or better performance
* The idea of FPN (Feature Pyramid Networks) is added to leverage the benefit from all the prior computations and fine-grained features early on in the network
* The bboxes anchors are defined using k-means (same as in YOLOv2), but YOLOv3 uses three prior boxes for different scales

## Model architecture

* **Darknet-53 classifier** - The network is a hybrid approach between the network used in YOLOv2 (Darknet-19), and the residual networks. It uses successive _3 × 3_ and _1 × 1_ convolutional layers but now has some shortcut connections as well and is significantly larger (53 layers)

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/fb6d5b7d-9d6d-4c93-8fa1-314e76f96b2d" alt="darknet_53" height="400"/>
</p>

* **Detector** - From the base feature extractor several convolutional layers are added:
	* The last of these predicts a _3-d_ tensor encoding bounding box, objectness, and class predictions. For COCO the predictions include 3 boxes at each scale so the tensor is _S × S × [3 ∗ (4 + 1 + 80)]_ for the 4 bbox offsets, 1 objectness prediction, and 80 class predictions
 	* Next the feature map from 2 layers previous is taken and is upsampled by 2x. The feature map from earlier point in the network is also taken and it is merged with the upsampled features using concatenation. This method allows to get more meaningful semantic information from the upsampled features and finer-grained information from the earlier feature map. On top of that a few more convolutional layers are added to process this combined feature map, and eventually predict a similar tensor (_S × S × [3 ∗ (4 + 1 + 80)]_), although now twice the size.
  	* The same design if performed one more time to predict boxes for the final scale. Thus the predictions for the 3rd scale benefit from all the prior computation as well as fine-grained features from early on in the network.

## Training details:
The network is trained similar to YOLOv2



# Gaussian YOLO
2019 | [paper](https://arxiv.org/pdf/1904.04620) | _Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving_

This paper proposes a method for improving the detection accuracy while supporting a real-time operation by modeling the bounding box (bbox) of YOLOv3, which is the most representative of one-stage detectors, with a Gaussian parameter and redesigning the loss function. In addition, this paper proposes a method for predicting the localization uncertainty that indicates the reliability of bbox. By using the predicted localization uncertainty during the detection process, the proposed schemes can significantly reduce the FP and increase the true positive (TP), thereby improving the accuracy.

One of the most critical problems of most conventional deep-learning based object detection algorithms is that, whereas the bbox coordinates (i.e., localization) of the detected object are known, the uncertainty of the bbox result is not. Thus, conventional object detectors cannot prevent mislocalizations (i.e., FPs) because they output the deterministic results of the bbox without information regarding the uncertainty. To overcome this problem, the present paper proposes a method for improving the detection accuracy by modeling the bbox coordinates of YOLOv3, which only outputs deterministic values, as the Gaussian parameters (i.e., the mean and variance), and redesigning the loss function of bbox. Through this Gaussian modeling, a localization uncertainty for a bbox regression task in YOLOv3 can be estimated. Furthermore, to further improve the detection accuracy, a method for reducing the FP and increasing the TP by utilizing the predicted localization uncertainty of bbox during the detection process is proposed.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/31dbff4e-2507-42a5-9a2a-8cef84be2aea" alt="gaussian_yolo_v3" height="400"/>
</p>

As shown in figure above, the prediction feature map of YOLOv3 has three prediction boxes per grid, where each prediction box consists of bbox coordinates (i.e., $t_x$, $t_y$ , $t_w$, and $t_h$), the objectness score, and class scores. YOLOv3 outputs the objectness (i.e., whether an object is present or not in the bbox) and class (i.e., the category of the object), as a score of between zero and one. An object is then detected based on the product of these two values. Unlike the objectness and class information, bbox coordinates are output as deterministic coordinate values instead of a score, and thus the confidence of the detected bbox is unknown. Moreover, the objectness score does not reflect the reliability of the bbox well. It therefore does not know how uncertain the result of bbox is. In contrast, the uncer tainty of bbox, which is predicted by the proposed method, serves as the bbox score, and can thus be used as an indicator of how uncertain the bbox is. In YOLOv3, bbox regression is to extract the bbox center information (i.e., $t_x$ and $t_y$ ) and bbox size information (i.e., $t_w$ and $t_h$). Because there is only one correct answer (i.e., the GT) for the bbox of an object, complex modeling is not required for predicting the localization uncertainty. I other words, the uncertainty of bbox can be modeled using each single Gaussian model of $t_x$, $t_y$ , $t_w$, and $t_h$. A single Gaussian model of output $y$ for a given test input $x$ whose output consists of Gaussian parameters is as follows:

$$p(y|x) = N (y; μ(x), Σ(x))$$

where $μ(x)$ and $Σ(x)$ are the mean and variance functions,
respectively.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f925db46-f5b6-44f4-8365-6d0ea90aa775" alt="gaussian_yolo_comps" height="300"/>
</p>

## Gaussian Modeling:

In Gaussian YOLOv3, the bbox coordinates are modeled as Gaussian parameters, consisting of mean ($µ$) and variance ($Σ$). The Gaussian parameters for each bbox coordinate ($t_x$, $t_y$, $t_w$, $t_h$) are processed as follows:
$$ \mu_{t_x} = \sigma(\hat{\mu}{t_x}), \Sigma{t_x} = \sigma(\hat{\Sigma}{t_x})  $$
$$ \mu{t_y} = \sigma(\hat{\mu}{t_y}),  \Sigma{t_y} = \sigma(\hat{\Sigma}{t_y})  $$
$$ \mu{t_w} = \hat{\mu}{t_w},          \Sigma{t_w} = \sigma(\hat{\Sigma}{t_w})  $$
$$ \mu{t_h} = \hat{\mu}{t_h},          \Sigma{t_h} = \sigma(\hat{\Sigma}_{t_h}) $$

where ( $\sigma(x) = \frac{1}{1 + e^{-x}}$ ) is the sigmoid function.

## Reconstruction of Loss Function:

The loss function for bbox regression is redesigned as a Negative Log Likelihood (NLL) loss to penalize the model based on the uncertainty of the predicted bbox coordinates:

$$L_x = - \sum_{i=1}^{W} \sum_{j=1}^{H} \sum_{k=1}^{K} \gamma_{ijk} \log(N(x^G_{ijk} | \mu_{t_x}(x_{ijk}), \Sigma_{t_x}(x_{ijk})) + \epsilon)$$

where $L_x$ is the NLL loss of $t_x$ coordinate and the others (i.e., $L_y$ , $L_w$, and $L_h$) are the same as $L_x$ except for each parameter. 

$$\gamma_{ijk} = \frac{\omega_{scale} \times \delta_{obj_{ijk}}}{2}$$
$$\omega_{scale} = 2 - w^G \times h^G$$

## Utilization of Localization Uncertainty:

The localization uncertainty, represented as the average of uncertainties of predicted bbox coordinates, is utilized during the detection process. The detection criterion for Gaussian YOLOv3 is defined as:

$$Cr. = \sigma(Object) \times \sigma(Class_i) \times (1 - Uncertainty_{aver})$$

where $\sigma(Object)$ is the objectness score, $\sigma(Class_i)$ is the score of the i-th class, and $Uncertainty_{aver}$ is the average uncertainty of bbox coordinates.

## Benefits:

* Improved Accuracy: By modeling bbox coordinates as Gaussian parameters, the algorithm can estimate localization uncertainty, leading to more accurate object detection.
* Robustness to Noisy Data: The redesigned loss function penalizes noisy data during training, making the model more robust and improving performance.
* Reduction in False Positives (FP): Utilizing localization uncertainty helps filter out predictions with high uncertainty, reducing false positives and improving detection precision.
* Increase in True Positives (TP): By considering localization uncertainty during detection, the algorithm can increase true positives, enhancing the overall detection performance and safety of autonomous vehicles.




# PP-YOLO
2020 | [paper](https://arxiv.org/pdf/2007.12099) | _PP-YOLO: An Effective and Efficient Implementation of Object Detector_

The goal of this paper is to implement an object detector with relatively balanced effectiveness and efficiency that can be directly applied in actual application scenarios, rather than propose a novel detection model. Considering that YOLOv3 has been widely used in practice, authors develop a new object detector based on YOLOv3. They mainly try to combine various existing tricks that almost not increase the number of model parameters and FLOPs, to achieve the goal of improving the accuracy of detector as much as possible while ensuring that the speed is almost unchanged.

An one-stage anchor-based detector is normally made up of a backbone network, a detection neck, which is typically a feature pyramid network (FPN), and a detection head for object classification and localization. They are also common components in most of the one-stage anchor-free detectors based on anchor-point. Authors first revise the detail structure of YOLOv3 and introduce a modified version which replace the backbone to ResNet50-vd-dcn, which is used as the basic baseline in this paper. Then a bunch of tricks is introduced which can improve the performance of YOLOv3 almost without losing efficiency.

## Architecture

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/05eb2b99-e48f-4f03-a7ee-d52a0ee201ba" alt="PP_YOLO" height="450"/>
</p>

**Backbone** - the overall architecture of YOLOv3 is shown in figure above. In original YOLOv3, DarkNet-53 is first applied to extract feature maps at different scales. Since ResNet has been widely used and and has been studied more extensively, there are more different variants for selection, and it has also been better optimized by deep learning frameworks. Authors decided to replace the original backbone DarkNet-53 with ResNet50-vd in PP-YOLO. Considering, that directly replacing DarkNet-53 with ResNet50-vd would hurt the performance of YOLOv3 detector, they replace some convolutional layers in ResNet50-vd with deformable convolutional layers. The effectiveness of Deformable Convolutional Networks (DCN) has been verified in many detection models. DCN itself will not significantly increase the number of parameters and FLOPs in the model, but in practical application, too many DCN layers will greatly increase inference time. Therefore, in order to balance the efficiency and effectiveness, only the $3 × 3$ convolution layers in the last stage were replaced with DCNs. This modified backbone is denoted as ResNet50-vd-dcn, and the output of stage 3, 4 and 5 as $C_3$, $C_4$, $C_5$.

**Detection Neck** - The FPN is used to build a feature pyramid with lateral connections between feature maps. Feature maps $C_3$, $C_4$, $C_5$ are input to the FPN module. Authors denote the output feature maps of pyramid level $l$ as $P_l$, where $l = 3, 4, 5$ in the experiments. The resolution of $P_l$ is $\frac{W}{2^l} \times \frac{H}{2^l}$ for an input image of size $W × H$. The detail structure of FPN is shown in figure above.

**Detection Head** - the detection head of YOLOv3 is very simple. It consists of two convolutional layers. A $3 × 3$ convolutional followed by an $1 × 1$ convolutional layer is adopt to get the final predictions. The output channel of each final prediction is $3(K + 5)$, where $K$ is number of classes. Each position on each final prediction map has been associate with three different anchors. For each anchor, the first K channels are the prediction of probability for $K$ classes. The following 4 channels are the prediction for bounding box localization. The last channel is the prediction of objectness score. For classification and localization, _Cross Entropy Loss_ and $L_1$ _Loss_ is adopted correspondingly. An _Objectness Loss_ is applied to supervise objectness score, which is used to identify whether is there an object or not.

### Selection of Tricks

The various tricks we used in this paper are described in this section. These tricks are all already existing, which coming from different works. This paper does not propose an novel detection method, but just focuses on combining the existing tricks to implement an effective and efficient detector. Because many tricks cannot be applied to YOLOv3 directly, authors needed to adjust them according to the its structure:

* **Larger Batch Size** - using a larger batch size can improve the stability of training and get better results. Authors change the training batch size from 64 to 192, and adjust the training schedule and learning rate accordingly

* **EMA** - when training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values. The Exponential Moving Average (EMA) compute the moving averages of trained parameters using exponential decay. For each parameter $W$ , a shadow parameter is maintained:

$$W_{EMA} = λW_{EMA} + (1 − λ)W$$

where $λ$ is the decay. Authors apply EMA with decay $λ$ of $0.9998$ and use the shadow parameter $W_{EMA}$ for evaluation.

* **DropBlock** - a form of structured dropout where units in a contiguous region of a feature map are dropped together. Different from the original paper, authors only apply DropBlock to the FPN, since they find that adding DropBlock to the backbone will lead to a decrease of the performance. The detailed inject points of the DropBlock are marked by ”triangles” in figure above.

* **IoU Loss** - bounding box regression is the crucial step in object detection. In YOLOv3, $L_1$ loss is adopted for bounding box regression. It is not tailored to the mAP evaluation metric, which is strongly rely on Intersection over Union (IoU). IoU loss and other variations such as CIoU loss and GIoU loss have been proposed to address this problem. Different from YOLOv4, authors do not replace the $L_1$-loss with IoU loss directly, but add another branch to calculate IoU loss. They find that the improvements of various IoU loss are similar, so they choose the most basic IoU loss.

* **IoU Aware** - in YOLOv3, the classification probability and objectness score is multiplied as the final detection confidence, which do not consider the localization accuracy. To solve this problem, an IoU prediction branch is added to measure the accuracy of localization. During training, IoU aware loss is adopted to training the IoU prediction branch. During inference, the predicted IoU is multiplied by the classification probability and objectiveness score to compute the final detection confidence, which is more correlated with the localization accuracy. The final detection confidence is then used as the input of the subsequent NMS.

* **Grid Sensitive** - an effective trick introduced by YOLOv4. When decoding the coordinate of the
bounding box center $x$ and $y$, in original YOLOv3, one can get them by

$$x = s · (g_x + σ(p_x))$$
$$y = s · (g_y + σ(p_y ))$$

where $σ$ is the sigmoid function, $g_x$ and $g_y$ are integers and $s$ is a scale factor. Obviously, $x$ and $y$ cannot be exactly equal to $s · g_x$ or $s · (g_x + 1)$. This makes it difficult to predict the centres of bounding boxes that just located on the grid boundary. It is possible to address this problem, by changing the equations to:

$$x = s · (g_x + α · σ(p_x) − (α − 1)/2)$$
$$y = s · (g_y + α · σ(p_y ) − (α − 1)/2)$$

where $α$ is set to 1.05 in this paper. This makes it easier for the model to predict bounding box center exactly located on the grid boundary.

* **Matrix NMS** - it is motivated by Soft-NMS, which decays the other detection scores as amonotonic decreasing function of their overlaps. However, such process is sequential like traditional Greedy NMS and could not be implemented in parallel. Matrix NMS views this process from another perspective and implement it in a parallel manner. Therefore, the Matrix NMS is faster than traditional NMS, which will not bring any loss of efficiency

* **CoordConv** - it works by giving convolution access to its own input coordinates through the use of extra coordinate channels. CoordConv allows networks to learn either complete translation invariance or varying degrees of translation dependence. Considering that CoordConv will add two inputs channels to the convolution layer, some parameters and FLOPs will be added. In order to reduce the loss of efficiency as much as possible, authors do not change convolutional layers in backbone, and only replace the $1 x 1$ convolution layer in FPN and the first convolution layer in detection head with CoordConv. The detailed inject points of the CoordConv are marked by ”diamonds” in figure above.

* **SPP** - Spatial Pyramid Pooling (SPP) integrates spatial pyramid matching into CNN and use _max-pooling_ operation instead of _bag-of-words_ operation. YOLOv4 apply SPP module by concatenating _max-pooling_ outputs with kernel size $k × k$, where $k = {1, 5, 9, 13}$, and $stride = 1$. Under this design, a relatively large $k × k$ _max-pooling_ effectively increase the receptive field of backbone feature. In detail, the SPP only applied on the top feature map as shown in figure above with ”star” mark. No parameter are introduced by SPP itself, but the number of input channel of the following convolutional layer will increase. 

* **Better Pretrained Model** - using a pretrained model with higher classification accuracy on ImageNet may result in better detection performance. Authors use the distilled ResNet50-vd model




# YOLO v4
2020 | [paper](https://arxiv.org/pdf/2004.10934.pdf) | _YOLOv4: Optimal Speed and Accuracy of Object Detection_

YOLOv4 addresses the need for real-time object detection systems that can achieve high accuracy while maintaining fast inference speeds and possibility to train the object detector on a single customer GPU. Authors of the paper put a great emphasis on analyzing and evaluating the possible training and architectural choices for object detector training and architecture. Two sets of concepts were evaluated separately, that is the Bag of Freebies ([**BoF**](#bag-of-freebies)) and Bag of Specials ([**BoS**](#bag-of-specials)). 

Authors noticed that the reference model which is optimal for classification is not always optimal for a detector. In contrast to the classifier, the detector requires the following:
* Higher input network size (resolution) – for detecting multiple small-sized objects
* More layers – for a higher receptive field to cover the increased size of input network
* More parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image
They also noticed that the influence of the receptive field with different sizes is summarized as follows:
* Up to the object size - allows viewing the entire object
* Up to network size - allows viewing the context around the object
* Exceeding the network size - increases the number of connections between the image point and the final activation

## How it works

Different settings of BoF and BoS were tested for two backbones, that is CSPResNext50 and CSPDarknet53 (CSP module applied to ResNext50 and Darknet53 architectures). The settings include:
* Activations: ReLU, leaky-ReLU, Swish, or Mish
* bbox regression loss: MSE, IoU, GIoU, CIoU, DIoU
* Data augmentation: CutOut, MixUp, CutMix, Mosaic, Self-Adversarial Training
* Regularization method: DropBlock
* Normalization of the network activations by their mean and variance: Batch Normalization ([BN](https://arxiv.org/pdf/1502.03167)), Cross-Iteration Batch Normalization ([CBN](https://arxiv.org/pdf/2002.05712))
* Skip-connections: Residual connections, Weighted residual connections, Multiinput weighted residual connections, or Cross stage partial connections (CSP)
* Eliminate grid sensitivity problem - equation used in YOLOv3 to calculate object coordinates had a flaw related to the need of very high absolute values predictions for points on grid. Authors solved this problem through multiplying the sigmoid by a factor exceeding 1.0, so eliminating the effect of grid on which the object is undetectable
* IoU threshold - using multiple anchors for a single ground truth IoU (truth, anchor) > IoU threshold
* Optimized Anchors - using the optimized anchors for training with the 512x512 network resolution

In order to make the designed detector more suitable for training on single GPU, authors made additional design and improvement as follows:
* Introduced a new method of data augmentation - Mosaic, and Self-Adversarial Training (SAT)
	* Mosaic - represents a new data augmentation method that mixes 4 training images. Thus 4 different contexts are while CutMix mixes only 2 input images. This allows detection of objects outside their normal context. In addition, batch normalization calculates activation statistics from 4 different images on each layer. This significantly reduces the need for a large mini-batch size
	* Self-Adversarial Training (SAT) - new data augmentation technique that operates in 2 forward backward stages. In the 1st stage the neural network alters the original image instead of the network weights. In this way the neural network executes an adversarial attack on itself, altering the original image to create the deception that there is no desired object on the image. In the 2nd stage, the neural network is trained to detect an object on this modified image in the normal way
* Selected optimal hyper-parameters while applying genetic algorithms
* Modified some exsiting methods to make YOLOv4 design suitble for efficient training and detection - modified SAM, modified PAN, and Cross mini-Batch Normalization (CmBN)

## Model architecture

YOLOv4 consists of:
* Backbone - CSPDarknet53
* Neck - SPP and PAN
* Head - YOLOv3 head

YOLOv4 Backbone uses:
* BoF: CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing
* BoS: Mish activation, Cross-stage partial connections (CSP), Multiinput weighted residual connections (MiWRC)

YOLOv4  Detector uses:
* BoF: CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid
sensitivity, Using multiple anchors for a single ground truth, [Cosine annealing scheduler](https://arxiv.org/pdf/1608.03983v5), Optimal hyperparameters, Random training shapes
* BoS: Mish activation, modified SPP-block, modified SAM-block, modified PAN path-aggregation block, DIoU-NMS

Modified SPP, PAN and PAN:
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/cf3549cd-b72f-4369-9c86-50feef5ffc42" alt="modified_SPP" height="300"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3aaee815-a819-41dd-9b0f-a11c0223af92" alt="modified_SAM" height="300"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/72e5c007-20ff-44e9-a42d-b90d0ee2e019" alt="modified_PAN" height="300"/>
</p>


## Training details

**Classification** (ImageNet): 
* training steps is 8,000,000
* batch size and the mini-batch size are 128 and 32, respectively
* polynomial decay learning rate scheduling strategy is adopted with initial learning rate 0.1
* warm-up steps is 1000
* the momentum and weight decay are respectively set as 0.9 and 0.005
* All BoS experiments use the same hyper-parameter as the default setting, and in the BoF experiments, additional 50% training steps are added
* BoF experiments: MixUp, CutMix, Mosaic, Bluring data augmentation, and label smoothing regularization methods
* BoS experiments: LReLU, Swish, and Mish activation function. All experiments are trained with a 1080 Ti or 2080 Ti GPU.

**Object detection** (MS COCO): 
* training steps is 500,500
* step decay learning rate scheduling strategy is adopted with initial learning rate 0.01 and multiply with a factor 0.1 at the 400,000 steps and the 450,000 steps, respectively
* momentum and weight decay are respectively set as 0.9 and 0.0005
* All architectures use a single GPU to execute multi-scale training in the batch size of 64
* mini-batch size is 8 or 4 depend on the architectures and GPU memory limitation
* Except for using genetic algorithm for hyper-parameter search experiments, all other experiments use default setting
* Genetic algorithm used YOLOv3-SPP to train with GIoU loss and search 300 epochs for min-val 5k sets. Searched settings are adopted:  learning rate 0.00261, momentum 0.949, IoU threshold for assigning ground truth 0.213, and loss normalizer 0.07 for genetic algorithm experiments
* For all experiments, only one GPU is used for training, so techniques such as syncBN that optimizes multiple GPUs are not used
* BoF experiments: grid sensitivity elimination, mosaic data augmentation, IoU threshold, genetic algorithm, class label smoothing, cross mini-batch normalization, self-adversarial training, cosine annealing scheduler, dynamic mini-batch size, DropBlock, Optimized Anchors, different kind of IoU losses
* BoS experiments: Mish, SPP, SAM, RFB, BiFPN, and [Gaussian YOLO](https://arxiv.org/pdf/1904.04620)



# YOLOv5
2020 | [code](https://github.com/ultralytics/yolov5), [docs](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/) | _YOLOv5_

YOLOv5 is built on top of the work of [_YOLO v3_](#yolo-v3) and [_YOLO v4_](#yolo-v4). All the YOLOv5 models are composed of the same 3 components: _CSP-Darknet53_ as a backbone, [_SPP_](#SPP) (**S**patial **P**yramid **P**ooling) and [_PANet_](#PAN) (**P**ath **A**ggregation **Net**works) in the model neck and the head used in YOLOv4. Most of the YOLOv5 blocks are improved using the [_CSP_](#CSP) (**C**ross **S**tage **P**artial) module.

The key changes in YOLOv5 that didn't exist in previous version are: applying the CSPNet to the Darknet53 backbone, the integration of the Focus layer (conv 6x6) to the CSP-Darknet53 backbone, replacing the SPP block by the SPPF block in the model neck and applying the CSPNet strategy on the PANet model. YOLOv5 and YOLOv4 tackled the problem of grid sensitivity and can now detect easily bounding boxes having center points in the edges. Finally, YOLOv5 is lighter and faster than previous versions.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d3563351-69f1-4a3c-ad5f-704d744b057b" alt="yolo_v5_short" height="350"/>
</p>


## How it works

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/821b0953-8822-4cfe-bde4-fdfcfc6f2fa1" alt="yolo_v5_brief" height="200"/>
</p>

* The image is processed through an input layer (input) and sent to the backbone for feature extraction.
* Backbone - creates feature maps of different sizes
* Neck - fuses these features through the feature fusion network to finally generate three feature maps P3, P4, and P5 (in the YOLOv5, the dimensions are expressed with the size of _80 × 80_, _40 × 40_ and _20 × 20_) to detect small, medium, and large objects in the picture, respectively.
* Head - three feature maps are sent to the prediction head, the confidence calculation and bbox regression are executed for each pixel in the feature map using the prior anchors, so as to obtain a multi-dimensional array (bboxes) including object class, class confidence, box coordinates, width, and height information.
* Postprocessing (NMS) - by setting the corresponding thresholds (confthreshold, objthreshold) to filter the useless information in the array, and performing a non-maximum suppression (NMS) process, the final detection is done


## Model architecture

An example of yolov5L architecture:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/c7660fce-8a0c-41b5-a429-1ca77d0f597c" alt="yolo_v5_L" height="600"/>
</p>

* **Bakcbone** - _CSP-Darknet53_ architecture based on Darknet53 from YOLOv3 to which the authors applied the Cross Stage Partial (_CSP_) network strategy.
YOLO is a deep network, it uses residual and dense blocks in order to enable the flow of information to the deepest layers and to overcome the vanishing gradient problem. However one of the perks of using dense and residual blocks is the problem of redundant gradients. _CSPNet_ helps tackling this problem by truncating the gradient flow. According to the authors of CSP:

> _CSP_ network preserves the advantage of _DenseNet's_ feature reuse characteristics and helps reducing the excessive amount of redundant gradient information by truncating the gradient flow.

_YOLOv5_ employs _CSPNet_ strategy to partition the feature map of the base layer into two parts and then merges them through a cross-stage hierarchy as shown in the figure bellow

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/e7bf01c8-a015-4f49-a216-e3549a4c3612" alt="yolo_v5_CSP" height="250"/>
</p>

Applying this strategy comes with big advantages to YOLOv5, since it helps reducing the number of parameters and helps reducing an important amount of computation (less FLOPS) which lead to increasing the inference speed that is crucial parameter in real-time object detection models

* **Neck** - YOLOv5 brought two major changes to the model neck. First a variant of Spatial Pyramid Pooling (_SPP_) has been used, and the Path Aggregation Network (_PANet_) has been modified by incorporating the _BottleNeckCSP_ in its architecture

_PANet_ is a feature pyramid network, it has been used in previous version of YOLO (YOLOv4) to improve information flow and to help in the proper localization of pixels in the task of mask prediction. In YOLOv5 this network has been modified by applying the CSPNet strategy to it as shown in the network's architecture figure

_SPP_ block performs an aggregation of the information that receives from the inputs and returns a fixed length output. Thus it has the advantage of significantly increasing the receptive field and segregating the most relevant context features without lowering the speed of the network. This block has been used in previous versions of YOLO (YOLOv3 and YOLOv4) to separate the most important features from the backbone, however in YOLOv5(6.0/6.1) SPPF has been used , which is just another variant of the SPP block, to improve the speed of the network (same outputs, but faster).

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8f95fa80-0184-48c5-97fb-55a5c08f123a" alt="yolo_v5_SPP" height="200"/>
</p>

* **Head** - _YOLOv5_ uses the same head as _YOLOv3_ and _YOLOv4_. It is composed from three convolution layers that predicts the location of the bounding boxes _(x, y, height, width)_, the scores and the objects classes. The equation to compute the target coordinates for the bounding boxes have changed from previous versions, the difference is shown in the figure bellow.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8d62b83e-0916-4a2c-a37f-ae3e731f5301" alt="yolo_v5_head_eq" height="150"/>
</p>

* **Activation** - for YOLOv5 the authors went with _SiLU_ and _Sigmoid_ activation function. _SiLU_ stands for Sigmoid Linear Unit and it is also called the swish activation function. It has been used with the convolution operations in the hidden layers. While the Sigmoid activation function has been used with the convolution operations in the output layer

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3e2142d1-4b9e-4816-b5d5-0aacb69a3bb9" alt="yolo_v5_activation" height="200"/>
</p>


## Loss function
YOLOv5 returns three outputs: the classes of the detected objects, their bounding boxes and the objectness scores. Thus, it uses _BCE_ (Binary Cross Entropy) to compute the classes loss and the objectness loss. While [_CIoU_](#ciou) (Complete Intersection over Union) loss to compute the location loss. The loss in YOLOv5 is computed as a combination of three individual loss components:

* Classes Loss (**BCE Loss**) - Binary Cross-Entropy loss, measures the error for the classification task.
* Objectness Loss (**BCE Loss**) - Another Binary Cross-Entropy loss, calculates the error in detecting whether an object is present in a particular grid cell or not.
* Location Loss (**CIoU Loss**) - Complete IoU loss, measures the error in localizing the object within the grid cell.

The formula for the final loss is given by the following equation

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/57d86a5a-2f08-4a34-a7bd-21fd39a9988b" alt="yolo_v5_loss" height="40"/>
</p>

The objectness losses of the three prediction layers (P3, P4, P5) are weighted differently. The balance weights are _[4.0, 1.0, 0.4]_ respectively. This approach ensures that the predictions at different scales contribute appropriately to the total loss.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d1aaf623-29bf-45dd-9574-7d103a260646" alt="yolo_v5_loss_balance" height="70"/>
</p>


## Other improvements

* **First conv layers** - replaced the three first layers of the network with single 6x6 convolution. It helped reducing the number of parameters, the number of FLOPS and the CUDA memory while improving the speed of the forward and backward passes with minor effects on the mAP.
* **Eliminating grid sensitivity** - It was hard for the previous versions of YOLO to detect bounding boxes on image corners mainly due to the equations used to predict the bounding boxes, but the new equations presented above helped solving this problem by expanding the range of the center point offset from [0, 1] to _[-0.5, 1.5]_ (left) therefore the offset can be easily 1 or 0 (coordinates can be in the image's edge) as shown in the image in the left. Also the height and width scaling ratios (right) were unbounded in the previous equations which may lead to training instabilities but now this problem has been reduced.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d49af842-4256-4ae9-83f4-754d2ef335fb" alt="yolo_v5_center" height="200"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/91d65749-93af-4eb7-827e-dc8b974f8fcf" alt="yolo_v5_hw" height="200"/>
</p>

* **Targets building** - the targets building process in YOLOv5 is critical for training efficiency and model accuracy. It involves assigning ground truth boxes to the appropriate grid cells in the output map and matching them with the appropriate anchor boxes. This process follows these steps:
    1. Calculate the ratio of the ground truth box dimensions and the dimensions of each anchor template
    2. If the calculated ratio is within the threshold, match the ground truth box with the corresponding anchor
    3. Assign the matched anchor to the appropriate cells, keeping in mind that due to the revised center point offset, a ground truth box can be assigned to more than one anchor. Because the center point offset range is adjusted from _[0, 1]_ to _[-0.5, 1.5]_. GT Box can be assigned to more anchors.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/96994899-f1f9-46fe-8ad6-e3f3071208b9" alt="yolo_v5_target_build_1" height="150"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/fce7cd74-24e4-4351-ac53-841f12137e9a" alt="yolo_v5_target_build_2" height="150"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/c65841ad-51c0-47f5-af45-fcfe9eee20bd" alt="yolo_v5_target_build_3" height="250"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/6164c3e2-4c92-4dbf-aeaf-34f1eb1326f9" alt="yolo_v5_target_build_4" height="200"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/e47185fd-3a7d-4896-b0ba-ecb2fbcf5a19" alt="yolo_v5_target_build_5" height="280"/>
</p>



## Training details

YOLOv5 applies several sophisticated training strategies to enhance the model's performance. They include
* **Multiscale Training** - the input images are randomly rescaled within a range of 0.5 to 1.5 times their original size during the training process
* **Warmup** and **Cosine LR Scheduler** - a method to adjust the learning rate to enhance model performance
* **AutoAnchor** - this strategy optimizes the prior anchor boxes to match the statistical characteristics of the ground truth boxes for custom data
* **Exponential Moving Average (EMA)** - A strategy that uses the average of parameters over past steps to stabilize the training process and reduce generalization error
* **Mixed Precision Training** - A method to perform operations in half-precision format, reducing memory usage and enhancing computational speed
* **Hyperparameter Evolution** - A strategy to automatically tune hyperparameters to achieve optimal performance.
* **Data augmentation**:
	* _Mosaic Augmentation_ - combines four training images into one in ways that encourage object detection
 	* _Copy-Paste Augmentation_ - copies random patches from an image and pastes them onto another randomly chosen image, effectively generating a new training sample models to better handle various object scales and translations
	* _Random Affine Transformations_ - this includes random rotation, scaling, translation, and shearing of the images
 	* _MixUp Augmentation_ - creates composite images by taking a linear combination of two images and their associated labels
	* _HSV Augmentation_ - random changes to the Hue, Saturation, and Value of the images
	* _Random Horizontal Flip_ - randomly flips images horizontally

Sources: [[1](https://sh-tsang.medium.com/brief-review-yolov5-for-object-detection-84cc6c6a0e3a)], [[2](https://iq.opengenus.org/yolov5/)], [[3](https://github.com/ultralytics/yolov5/issues/6998#1)]




# Scaled YOLO v4
2021 | [paper](https://arxiv.org/pdf/2011.08036.pdf) | _Scaled-YOLOv4: Scaling Cross Stage Partial Network_
Scaled version of YOLOv4 built on top of the [YOLOv4](#yolo-v4). Authors have shown that the YOLOv4 object detection neural network based on the [CSP](#csp) approach, scales both up and down and is applicable to small and large networks while maintaining optimal speed and accuracy. They proposed a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network.


## How it works

When designing the efficient model scaling methods, authors main principles are:
* when the scale is up, the lower the quantitative cost we want to increase, the better
* when the scale is down, the higher the quantitative cost we want to decrease, the better

The [CSP](https://arxiv.org/pdf/1911.11929) can be applied to various CNN architectures, while reducing the amount of parameters and computations. In addition, it also improves accuracy and reduces inference time. CSP connection is extremely efficient, simple, and can be applied to any neural network. The idea is:
* half of the output signal goes along the main path (generates more semantic information with a large receiving field)
* and the other half of the signal goes bypass (preserves more spatial information with a small perceiving field)

Due to that, authors used CSP-ized models to perform model scaling. 

In case of the _tiny_ model, authors wanted the computations to be even smaller, so they adapted the [OSANet](https://arxiv.org/pdf/1904.09730) upgraded by [CSP](https://arxiv.org/pdf/1911.11929) module to create the CSPOSANet backbone

Usually, it is possible to adjust the scaling factors of an object detector’s input, backbone, and neck. The potential scaling factors that can be adjusted are summarized:
* Input: _size_
* Backbone: _width_, _depth_, _#stage_ (number of stages)
* Neck: _width_, _depth_, _#stage_ (number of stages)

The biggest difference between image classification and object detection is that the former only needs to identify the category of the largest component in an image, while the latter needs to predict the position and size of each object in an image. In one-stage object detector, the feature vector corresponding to each location is used to predict the category and size of an object at that location. The ability to better predict the size of an object basically depends on the receptive field of the feature vector. In the CNN architecture, the thing that is most directly related to receptive field is the stage, and the feature pyramid network (FPN) architecture tells us that higher stages are more suitable for predicting large objects. When the input image size is increased, if one wants to have a better prediction effect for large objects, he/she must increase the depth or number of stages of the network. When performing scaling up, authors first perform compound scaling on input size, and number of stages, and then according to real-time requirements, we further perform scaling on depth and width respectively

 
## Model architecture

* **Backbone** - in the design of CSPDarknet53, the computation of down-sampling convolution for cross-stage process is not included in a residual block. Therefore, we can deduce that CSPDarknet stage will have a better computational advantage over Darknet stage only when number of layers in a block is greater than 1. The number of residual layer owned by each stage in CSPDarknet53 is 1-2-8-8-4 respectively. In order to get a better speed/accuracy trade-off, the first CSP stage is converted into original Darknet residual layer
* **Neck** - in order to effectively reduce the amount of computation, [CSP](https://arxiv.org/pdf/1911.11929) is applied to the [PAN](https://arxiv.org/pdf/1803.01534) architecture in YOLOv4. The computation list of a PAN architecture is illustrated below (left). It mainly integrates the features coming from different feature pyramids, and then passes through two sets of reversed Darknet residual layer without shortcut connections. After applying [CSP](https://arxiv.org/pdf/1911.11929) (right), the computation is cut down by 40% and the neck is called CSPPAN.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/dfa8d3b9-34d6-4244-a370-a3f2834f9037" alt="reversed_CSP_dark_layers_SPP" height="300"/>
</p>

* **SPP** - the [SPP](https://arxiv.org/pdf/1406.4729) module was originally inserted in the middle position of the first computation list group of the neck. Therefore, in Scaled YOLOv4, the SPP is also inserted in the middle position of the first computation list group of the CSPPAN

Improvements in Scaled YOLOv4 over YOLOv4:
* Improved network architecture - backbone is optimized and Neck (PAN) uses CSP connections and Mish activation
* Exponential Moving Average (EMA) is used during training — this is a special case of Stochastic Weight Averaging ([SWA](https://arxiv.org/pdf/1803.05407), [SWA-PyTorch](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/))
* Multiresolution - For each resolution of the network, a separate neural network is trained (in YOLOv4, only one neural network was trained for all resolutions)
* Improved objectness normalizers in prediction layers
* Changed activations for Width and Height, which allows faster network training
* Pre Processing - the Letter Box pre processing is used (?) to keep the aspect ratio of the input image (tiny version doesnt use Letter Box)


### YOLOv4-tiny

YOLOv4-tiny is designed for low-end GPU device. The _CSPOSANet_ with _PCB_ (Partial in Computational Block) architecture is used as backbone. Growth rate (_g_) is set to _b/2_  (_b_ - base width) and it grows to _b/2 + kg = 2b_ at the end (_k_ - number of layers). Through calculation,  _k = 3_ is set, and its architecture and the computational block of YOLOv4-tiny is shown below. As for the number of channels of each stage and the part of neck, the design of YOLOv3-tiny is followed.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/81da6cf6-6ce8-4e41-a5d4-aa47038c4b17" alt="yolo_v4_tiny" height="300"/>
</p>

#### YOLOv4-large
YOLOv4-large is designed for cloud GPU, the main purpose is to achieve high accuracy for object detection. Authors designed a fully CSP-ized model YOLOv4-P5 and scaled it up to YOLOv4-P6 and YOLOv4-P7. The structure of YOLOv4-P5, YOLOv4-P6, and YOLOv4-P7 is shown below. The compound scaling of input size and number of stages was performed. The depth scale of each stage was set to to 2^dsi , and ds to [1, 3, 15, 15, 7, 7, 7]. Finally, the width scaling was performed with the use of inference time as constraint.



<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/9c5e0666-d82d-4944-8877-889783e02090" alt="yolo_v4_large" height="400"/>
</p>


## Training details

The models are trained directly on MS COCO dataset (no pre-training on ImageNet). The training choices include:
* SGD optimizer
* Training time:
	* YOLOv4-tiny: 600 epochs,
	* YOLOv4-CSP: 300 epochs, 
	* YOLOv4-large: 300 epochs first and then followed by using stronger data augmentation method to train 150 epochs
* Hyperparameters (anchors, learning rate, degree of data augmentation) were found using k-means and genetic algorithms


## Loss functions

There are different Losses in YOLOv3, YOLOv4 and Scaled-YOLOv4:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/7005d806-0928-4686-8dfa-cb79252a5ee4" alt="scaled_yolo_v4_losses" height="250"/>
</p>

* for _bx_ and _by_ — this eliminates grid sensitivity in the same way as in YOLOv4, but more aggressively
* for _bw_ and _bh_ — this limits the size of the bounded-box to _4*Anchor_size_




# YOLO R
2021 | [paper](https://arxiv.org/pdf/2105.04206) | _You Only Learn One Representation: Unified Network for Multiple Tasks_
TODO



# YOLO X
2021 | [paper](https://arxiv.org/pdf/2107.08430.pdf) | _YOLOX: Exceeding YOLO Series in 2021_
TODO



# PP-YOLOE
2022 | [paper](http://arxiv.org/pdf/2203.16250v3) | _PP-YOLOE: An evolved version of YOLO_
TODO 



# RTMDet
2022 | [paper](http://arxiv.org/pdf/2212.07784v2) | _RTMDet: An Empirical Study of Designing Real-Time Object Detectors_




# YOLO v6
2022 | [paper](https://arxiv.org/pdf/2209.02976) | _YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications_
TODO



# YOLO v7
2022 | [paper](https://arxiv.org/pdf/2207.02696.pdf) | _YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors_
TODO



# YOLO v6.3
2023 | [paper](https://arxiv.org/pdf/2301.05586.pdf) | _YOLOv6 v3.0: A Full-Scale Reloading_
TODO



# PP-YOLOE-R
2022 | [paper](https://arxiv.org/pdf/2211.02386) | _PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector_



# DAMA-YOLO
2023 | [paper](http://arxiv.org/pdf/2211.15444v4) | _DAMO-YOLO : A Report on Real-Time Object Detection Design_
TODO



# Dynamic YOLO
2023 | [paper](https://arxiv.org/pdf/2304.05552) | _DynamicDet: A Unified Dynamic Architecture for Object Detection_
TODO



# YOLO Review
2023 | [paper](https://arxiv.org/pdf/2304.00501.pdf) | _A comprehensive review of YOLO: from YOLOv1 and beyond_
TODO



# YOLOv8
2023 | [paper](https://github.com/ultralytics/ultralytics) | _YOLOv8_
TODO



# YOLO v9
2023 | [paper](https://arxiv.org/pdf/2402.13616) | _YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information_
TODO


