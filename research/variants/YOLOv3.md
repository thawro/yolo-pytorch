
# YOLOv3

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
