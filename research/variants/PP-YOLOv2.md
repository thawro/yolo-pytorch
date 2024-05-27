# PP-YOLOv2

2021 | [paper](https://arxiv.org/pdf/2104.10419) | _PP-YOLOv2: A Practical Object Detector_

Being effective and efficient is essential to an object detector for practical use. To meet these two concerns, authors comprehensively evaluate a collection of existing refinements to improve the performance of PP-YOLO while almost keep the infer time unchanged. This paper analyzes a collection of refinements and empirically evaluates their impact on the final model performance through incremental ablation study.

## Revisit PP-YOLO

The implementation of the baseline model:

* **Pre-Processing** -  Apply:
    1. _Mixup_ training with a weight sampled from $Beta(α, β)$ distribution where $α = 1.5, β = 1.5$.
    2. _RandomColorDistortion_, _RandomExpand_, _RandCrop_ and _RandomFlip_ are applied one by one with probability 0.5
    3. Normalize RGB channels by subtracting $0.485, 0.456, 0.406$ and dividing by $0.229, 0.224, 0.225$, respectively
    4. The input size is evenly drawn from $[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]$.

* **Baseline Model** - PP-YOLOv2 baseline model is PP-YOLO which is an enhanced version of YOLOv3. Specifically, it first replaces the backbone to ResNet50-vd. After that a total of 10 tricks which can improve the performance of YOLOv3 almost without losing efficiency are added to YOLOv3 such
as _Deformable Conv_, _SSLD_, _CoordConv_, _DropBlock_, _SPP_ and so on. The architecture of PP-YOLO is presented in the previous paper (PP-YOLO).

* **Training Schedule** - On COCO train2017, the network is trained with stochastic gradient descent (SGD) for 500K iterations with a minibatch of 96 images distributed on 8 GPUs. The learning rate is linearly increased from 0 to 0.005 in 4K iterations, and it is divided by 10 at iteration 400K and 450K, respectively. Weight decay is set as 0.0005, and momentum is set as 0.9. Gradient clipping is adopted to stable the training procedure.

## Selection of Refinements

pp-yolov2

**Path Aggregation Network** - detecting objects at differ ent scales is a fundamental challenge in object detection. In practice, a detection neck is developed for building high-level semantic feature maps at all scales. In PP-YOLO, FPN is adopted to compose bottom-up paths. Recently, several FPN variants have been proposed to enhance the ability of pyramid representation. For example, BiFPN, PAN, RFP and so on. PP-YOLOv2 follows the design of PAN to aggregate the top-down information. The detailed structure of PAN is shown in figure above.

**Mish Activation Function** - _Mish_ has been proved effective in many practical detectors, such as YOLOv4 and YOLOv5. They adopt the mish activation function in the backbone. However authors of PP-YOLOv2 decided to apply _Mish_ only in detection neck, since they already had a powerful pre-trained backbone model (without Mish) and didn't want to train a new one.

**Larger Input Size** - increasing the input size enlarges the area of objects. Thus, information of the objects on a small scale will be preserved easier than before. As a result, performance will be increased. However, a larger input size occupies more memory. To apply this trick, one need to decrease batch size. To be more specific, authors reduced the batch size from 24 images per GPU to 12 images per GPU and expanded the largest input size from 608 to 768. The input size is evenly drawn from $[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]$.

**IoU Aware Branch** - in PP-YOLO, IoU aware loss is calculated in a soft weight format which is inconsistent with the original intention. Therefore, authors applied a soft label format. Here is the IoU aware loss:

$$ loss = −t ∗ log(σ(p)) − (1 − t) ∗ log(1 − σ(p)) $$

where $t$ indicates the IoU between the anchor and its matched ground-truth bounding box, $p$ is the raw output of IoU aware branch, $σ(·)$ refers to the sigmoid activation function. IoU aware loss is computed only for positive samples. By replacing the loss function, IoU aware branch works better than in PP-YOLO.
