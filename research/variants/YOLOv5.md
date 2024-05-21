
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
