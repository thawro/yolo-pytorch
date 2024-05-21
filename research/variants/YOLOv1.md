# YOLOv1

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