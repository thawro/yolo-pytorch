# DIoU

2018 | [paper](https://arxiv.org/pdf/1911.08287.pdf) | _Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression_

Intersection over Union (IoU) is the most popular metric for object detection:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/cdb085ad-22ab-4f06-8bea-433f4ae96177" alt="iou" height="60"/>
</p>

where $B_{gt} = (x^{gt}, y^{gt}, w^{gt}, h^{gt})$ is the ground-truth, and
$B = (x, y, w, h)$ is the predicted box. Conventionally, $l_n$-norm (e.g., $n = 1$ or $n = 2$) loss is adopted on the coordinates of $B$ and $B_{gt}$ to measure the distance between bounding boxes. However, $l_n$-norm loss is not a suitable choice to obtain the optimal IoU metric. In earlier works, the IoU loss was suggested to be adopted for improving the IoU metric:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/eab21c19-7365-4626-adc2-755f5cf0087c" alt="iou_loss" height="60"/>
</p>

However, IoU loss only works when the bounding boxes have overlap, and would not provide any moving gradient for non-overlapping cases. And then generalized IoU loss
(GIoU) is proposed by adding a penalty term:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/8435920f-e666-41c8-8eac-5c933a11178b" alt="giou_loss" height="60"/>
</p>

where $C$ is the smallest box covering $B$ and $B_{gt}$. Due to the introduction of penalty term, the predicted box will move towards the target box in non-overlapping cases. Although GIoU can relieve the gradient vanishing problem for non-overlapping cases, it still has several limitations. As shown below:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/51db5636-3187-4b90-8e68-2dfb450a23e0" alt="giou_vs_diou" height="300"/>
</p>

one can see that GIoU loss intends to increase the size of predicted box at first, making it have overlap with target box, and then the IoU term in GIoU equation will work to maximize the overlap area of bounding box. And from figure below:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/9298cee5-1c55-4549-956d-8b9899a73803" alt="iou_vs_giou_vs_diou" height="400"/>
</p>

GIoU loss will totally degrade to IoU loss for enclosing bounding boxes. Due to heavily relying on the IoU term, GIoU empirically needs more iterations to converge, especially for horizontal and vertical bounding boxes. Usually GIoU loss cannot well converge in the SoTA detection algorithms, yielding inaccurate detection.

To sum up, IoU loss converges to bad solutions for nonoverlapping cases, while GIoU loss is with slow convergence especially for the boxes at horizontal and vertical orientations. And when incorporating into object detection pipeline, both IoU and GIoU losses cannot guarantee the accuracy of regression

**Distance-IoU (DIoU)**
In this paper, authors proposed a Distance-IoU (DIoU) loss for bounding box regression. In particular, they simply added a penalty term on IoU loss to directly minimize the normalized distance between central points of two bounding boxes, leading to much faster convergence than GIoU loss. From figure above it can be seen, that DIoU loss can be deployed to directly minimize the distance between two bounding boxes:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/1c60033c-eac0-481e-afed-f47936cc5286" alt="diou" height="60"/>
</p>

where $b$ and $b_{gt}$ denote the central points of $B$ and $B_{gt}$, $ρ(·)$ is the Euclidean distance, and c is the diagonal length of the smallest enclosing box covering the two boxes. 

**Comparison with IoU and GIoU losses** -  The proposed DIoU loss inherits some properties from IoU and GIoU loss:

1. DIoU loss is still invariant to the scale of regression problem.
2. Similar to GIoU loss, DIoU loss can provide moving directions for bounding boxes when non-overlapping with target box.
3. When two bounding boxes perfectly match, $L_{IoU} = L_{GIoU} = L_{DIoU} = 0$. When two boxes are far away, $L_{GIoU} = L_{DIoU} → 2$.

And DIoU loss has several merits over IoU loss and GIoU loss, which can be evaluated by simulation experiment.

**Complete IoU Loss (CIoU)**
A good loss for bounding box regression should consider three important geometric factors, i.e., overlap area, central point distance and aspect ratio. By uniting the coordinates, IoU loss considers the overlap area, and GIoU loss heavily relies on IoU loss. The proposed DIoU loss aims at considering simultaneously the overlap area and central point distance of bounding boxes. However, the consistency of aspect ratios for bounding boxes is also an important geometric factor. Therefore, based on DIoU loss, the CIoU loss is proposed by imposing the consistency of aspect ratio:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/1413b442-6736-4a7f-b8eb-df968287a0fd" alt="ciou_loss" height="60"/>
</p>

where $α$ is a positive trade-off parameter, and $v$ measures the consistency of aspect ratio:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/f38a6af0-6008-40cd-96da-eea27fe2e568" alt="ciou_alpha" height="60"/>
</p>
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2e8976f5-3792-4cd1-a73b-db5b96fdcc78" alt="ciou_v" height="60"/>
</p>

> **_NOTE:_** Most of the new YOLO approaches use **DIoU** or **CIoU** as loss functions for bounding box regression.
