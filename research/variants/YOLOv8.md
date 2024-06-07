# YOLOv8

2023 | [paper](https://github.com/ultralytics/ultralytics) | _YOLOv8_

YOLOv8 is the latest iteration of the You Only Look Once (YOLO) family of object detection models, known for their speed and accuracy. Developed by the Ultralytics team, YOLOv8 builds upon the success of its predecessors while introducing several key innovations that push the boundaries of real-time object detection.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f293c08c-d5b7-4cd4-91fe-fe3e5bc0ba50" alt="yolo_v8" height="800"/>
</p>

## Key innovations

### Anchor-free

YOLOv8 is an anchor-free model. This means it predicts directly the center of an object instead of the offset from a known anchor box. Anchor boxes were a notoriously tricky part of earlier YOLO models, since they may represent the distribution of the target benchmark's boxes but not the distribution of the custom dataset. Anchor free detection reduces the number of box predictions, which speeds up Non-Maximum Suppression (NMS), a complicated post processing step that sifts through candidate detections after inference

### New Convolutions

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/836f9a07-6c1c-4c37-84de-dfc8fb62e0c4" alt="yolo_v8_c2f" height="350"/>
</p>

The stem's first $6 × 6$ _conv_ is replaced by a $3 × 3$, the main building block was changed, and **_C2f_** replaced **_C3_**. The module is summarized in the figure above, where $f$ is the number of features, $e$ is the expansion rate and $CBS$ is a block composed of a $Conv$, a $BatchNorm$ and a $SiLU$ layer. In C2f, all the outputs from the Bottleneck (fancy name for two $3 × 3$ _convs_ with residual connections) are concatenated. While in C3 only the output of the last Bottleneck was used.

The Bottleneck is the same as in YOLOv5 but the first _conv_'s kernel size was changed from $1 × 1$ to $3 × 3$. From this information, we can see that YOLOv8 is starting to revert to the ResNet block defined in 2015.

In the neck, features are concatenated directly without forcing the same channel dimensions. This reduces the parameters count and the overall size of the tensors

### Closing the Mosaic Augmentation

Deep learning research tends to focus on model architecture, but the training routine in YOLOv5 and YOLOv8 is an essential part of their success.

YOLOv8 augments images during training online. At each epoch, the model sees a slightly different variation of the images it has been provided.

One of those augmentations is called mosaic augmentation. This involves stitching four images together, forcing the model to learn objects in new locations, in partial occlusion, and against different surrounding pixels. 

However, this augmentation is empirically shown to degrade performance if performed through the whole training routine. It is advantageous to turn it off for the last ten training epochs.

This sort of change is exemplary of the careful attention YOLO modeling has been given in overtime in the YOLOv5 repo and in the YOLOv8 research.

### Other

* **Spatial Attention** - YOLOv8 incorporates a spatial attention mechanism that focuses on relevant parts of the image, leading to more precise object localization
* **Feature Fusion** - The C2f module effectively combines high-level semantic features with low-level spatial information, improving detection accuracy for small objects
* **Bottlenecks and SPPF** - Bottlenecks in the CSPDarknet53 backbone reduce computational complexity while maintaining accuracy. Additionally, the **_Spatial Pyramid Pooling Fast (SPPF)_** layer captures features at multiple scales, further enhancing detection performance
* **Mixed Precision Training** - Mixed precision training further enhances training speed and efficiency

## Architecture

YOLOv8 boasts several architectural updates targeted at boosting both accuracy and speed. These include:

* **Stem tweak** - Reducing the first _convolutional_ kernel size from $6 × 6$ to $3 × 3$ for efficient image abstraction.
* **Backbone bottleneck tweak** - Upscaling the first _convolutional_ kernel in the bottleneck area from $1 × 1$ to $3 × 3$ for better feature extraction.
* **Backbone building block swap** - Replacing the **_C3_** block from YOLOv5 with a new, more efficient design.
* **Anchor-free head** - Implementing a novel anchor-free head for object detection, eliminating the need for predefined anchor boxes.
* **New loss function** - Utilizing a modified loss function focusing on both bounding box location and classification confidence.

### Backbone

The backbone network is the foundation of YOLOv8, responsible for feature extraction from the input image. YOLOv8 employs **_CSPDarknet53_**, a variant of Darknet, as its backbone. The CSPDarknet53 architecture introduces a novel **_Cross-Stage Partial (CSP)_** connection, enhancing the information flow between different stages of the network and improving gradient flow during training

### Neck

The neck, also known as the feature extractor, merges feature maps from different stages of the backbone to capture information at various scales. YOLOv8 introduces a **_Path Aggregation Network (PANet)_** along with a novel **_C2f_** module as the neck structure. PANet facilitates information flow across different spatial resolutions, enabling the model to capture multi-scale features effectively. This module combines high-level semantic features with low-level spatial information, leading to improved detection accuracy, especially for small objects.

### Head

The head structure consists of multiple detection heads, each responsible for predicting _bounding boxes_, _class probabilities_, and _objectness scores_ at different scales. It utilizes a modified version of the YOLO head, incorporating dynamic anchor assignment and a novel IoU loss function.

## Training

YOLOv8 adopts a comprehensive training strategy to optimize its performance. One notable feature is the use of multiple training resolutions, allowing the model to learn from images at different scales.

Additionally, the model utilizes a mosaic data augmentation technique, combining multiple images into a single training input. This approach enhances the model’s ability to generalize to diverse scenarios and improves its robustness
