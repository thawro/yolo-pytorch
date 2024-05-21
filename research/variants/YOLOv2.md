# YOLOv2

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
