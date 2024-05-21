# Soft NMS

2017 | [paper](https://arxiv.org/pdf/1704.04503.pdf) | _Improving Object Detection With One Line of Code_

Non-maximum suppression (**NMS**) is an integral part of the object detection pipeline. First, it sorts all detection boxes on the basis of their scores. The detection box $M$ with the maximum score is selected and all other detection boxes with a significant overlap (using a pre-defined threshold) with $M$ are suppressed. This process is recursively applied on the remaining boxes. As per the design of the algorithm, if an object lies within the predefined overlap threshold, it leads to a miss. 

Authors proposed an improved version of NMS, that is a _**Soft-NMS**_, an algorithm which decays the detection scores of all other objects as a continuous function of their overlap with _M_. Hence, no object is eliminated in this process.

During the analysis of old NMS, authors wanted the improved metric to take the following conditions into account:

* Score of neighboring detections should be decreased to an extent that they have a smaller likelihood of increasing the false positive rate, while being above obvious false positives in the ranked list of detections.
* Removing neighboring detections altogether with a low NMS threshold would be sub-optimal and would increase the miss-rate when evaluation is performed at high overlap thresholds.
* Average precision measured over a range of overlap thresholds would drop when a high NMS threshold is used

It would be ideal if the penalty function was continuous, otherwise it could lead to abrupt changes to the ranked list of detections. A continuous penalty function should have no penalty when there is no overlap and very high penalty at a high overlap. Also, when the overlap is low, it should increase the penalty gradually, as $M$ should not affect the scores of boxes which have a very low overlap with it. However, when overlap of a box $b_i$ with $M$ becomes close to one, $b_i$ should be significantly penalized. Taking this into consideration, authors proposed a Gaussian penalty function:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/24ac7cb8-9152-49b3-9127-b13b03f98bf7" alt="soft_NMS_eq" height="60"/>
</p>

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/db925ea6-ff65-4af1-9be3-023f1b4f5a0f" alt="soft_NMS_algo" height="350"/>
</p>

This update rule (top) is applied in each iteration and scores of all remaining detection boxes are updated

> **_NOTE:_** Some of the YOLO approaches use the idea of **Soft-NMS** for post processing. Soft-NMS is improved later with the use of DIoU.