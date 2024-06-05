# YOLO-Pose

2022 | [paper](https://arxiv.org/pdf/2204.06806) | _YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss_

YOLO-pose is a novel heatmap-free approach for joint detection, and 2D multi-person pose estimation in an image based on the popular YOLO object detection framework. Existing heatmap based two-stage approaches are sub-optimal as they are not end-to-end trainable and training relies on a surrogate $L_1$ loss that is not equivalent to maximizing the evaluation metric, i.e. Object Keypoint Similarity (OKS). YOLO-pose framework allows to train the model end-to-end and optimize the OKS metric itself. The proposed model learns to jointly detect bounding boxes for multiple persons and their corresponding 2D poses in a single forward pass and thus bringing in the best of both top-down and bottom-up approaches. Proposed approach doesn’t require the post-processing of bottom-up approaches to group detected keypoints into a skeleton as each bounding box has an associated pose, resulting in an inherent grouping of the keypoints. Unlike top-down approaches, multiple forward passes are done away with since all persons are localized along with their pose in a single inference. YOLO-pose achieves new SoTA results on COCO validation (90.2% AP50) and test-dev set (90.3% AP50), surpassing all existing bottom-up approaches in a single forward pass without flip test, multi-scale testing, or any other test time augmentation.

Motivation for this work is to solve the problem of pose estimation without heatmaps and in line with object detection since challenges in object detection are similar to that of pose estimation, e.g. scale variation, occlusion, non-rigidity of the human body, and so on. Hence, if a person detector network can tackle these challenges, it can handle pose estimation as well. E.g. latest object detection frameworks try to mitigate the problem of scale variation by making predictions at multiple scales. YOLO-pose adopts the same strategy to predict human pose at multiple scales corresponding to each detection. Similarly, all major progress in the field of object detection are seamlessly passed on to the problem of pose estimation. The proposed pose estimation technique can be easily integrated into any computer vision system that runs object detection with almost zero increase in compute.

Authors call this approach YOLO-Pose, based on the YOLOv5 framework. This is the first focused attempt to solve the problem of 2D pose estimation without heatmap and with an emphasis on getting rid of various non-standardized post-processings that are currently used. YOLO-pose uses the same post-processing as in object detection. Authors validate models on the challenging COCO keypoint dataset and demonstrate competitive accuracy in terms of AP and significant improvement in terms of AP50 over models with similar complexity.

In YOLO-pose approach, an anchor box or an anchor point that matches the ground truth box stores its full 2d pose along with the bounding box location. Two similar joints from different persons can be close to each other spatially. Using a heatmap, it is difficult to distinguish two spatially close similar joints from different persons. However, if these two persons are matched against different anchors, it’s easy to distinguish spatially close and similar keypoints. Again, Keypoints associated with an anchor are already grouped. Hence, no need for any further grouping. In bottom-up approaches, keypoints of one person can be easily mistaken for another whereas YOLO-pose inherently tackles this problem. Unlike, top-down approaches, the complexity of YOLO-Pose is independent of the number of persons in an image. Thus, it has the best of both top-down and bottom-up approaches: Constant run-time as well as simple post-processing. Overall, YOLO-pose makes the following contributions:

* Propose solving multi-person pose estimation in line with object detection since major challenges like scale variation and occlusion are common to both. Thus, taking the first step toward unifying these two fields

* The heatmap-free approach uses standard object detection post-processing instead of complex post-processing involving Pixel level NMS, adjustment, refinement, line-integral, and various grouping algorithms. The approach is robust because of end-to-end training without independent post-processing.

* Extended the idea of IoU loss from box detection to keypoints. Object keypoint similarity (OKS) is not just used for evaluation but as a loss for training. OKS loss is scale-invariant and inherently gives different weighting to different keypoints

* Propose a joint detection and pose estimation framework. Pose estimation comes almost free with an object detection network.

YOLO-pose is a single shot approach like other bottom-up approaches. However, it doesn’t use heatmaps. Rather, it associates all keypoints of a person with anchors. It is based on YOLOv5 object detection framework and can be extended to other frameworks as well. Authors have validated it on YOLOX as well to a limited extent. Figure below illustrates the overall architecture with keypoint heads for pose estimation.

## Overview

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/ac2e330f-2182-47f9-a93d-1c71edcb526e" alt="yolo_pose" height="450"/>
</p>

In order to showcase the full potential of YOLO-pose, authors had to select an architecture that is good at detecting humans. YOLOv5 is a leading detector in terms of accuracy and complexity. Hence, it was selected as the base. YOLOv5 mainly focuses on 80 class COCO object detection with the box head predicting 85 elements per anchor. They correspond to _bounding box_, _object score_, and _confidence score_ for 80 classes. Corresponding to each grid location, there are three anchors of different shapes.

For human pose estimation, it boils down to a single class person detection problem with each person having 17 associated keypoints, and each keypoint is again identified with a _location_ and _confidence_: $\{x, y, conf\}$. So, in total there are 51 elements for 17 keypoints associated with an anchor. Therefore, for each anchor, the keypoint head predicts 51 elements and the box head predicts six elements. For an anchor with $n$ keypoints, the overall prediction vector is defined as:

$$ P_v = \{ C_x, C_y, W, H, box_{conf}, class_{conf}, K_x^1, K_y^1, K_{conf}^1, ..., K_x^n, K_y^n, K_{conf}^n \} $$

Keypoint confidence is trained based on the visibility flag of that keypoint. If a keypoint is either visible or occluded, then the ground truth confidence is set to 1 else if it is outside the field of view, confidence is set to zero. During inference, authors retain keypoints with confidence greater than 0.5. All other predicted keypoints are rejected. Predicted keypoint confidence is not used for evaluation. However, since the network predicts all 17 keypoints for each detection, it is needed to filter out keypoints that are outside the field of view. Else, there will be dangling keypoints resulting in a deformed skeleton. Existing bottom-up approaches based on heatmap don’t require this since keypoints outside the field of view are not detected in the first place.

YOLO-Pose uses CSP-darknet53 as backbone and PANet for fusing features of various scales from the backbone. This follows by four detection heads at different scales. Finally, there are two decoupled heads for predicting boxes and keypoints. In this work, authors limit the complexity to 150 GMACS and within that, they are able to achieve competitive results. With further increase in complexity, it is possible to further bridge the gap with top-down approaches. However, authors don’t pursue that path as the focus is on real-time models.

## Anchor based multi-person pose formulation

For a given image, an anchor that is matched against a person stores its entire 2D pose along with bounding box. The box coordinates are transformed w.r.t the anchor center whereas box dimensions are normalized against height and width of the anchor. Similarly, keypoint locations are transformed w.r.t to anchor center. However, Keypoints aren’t normalized with anchor height and width. Both Key-point and box are predicted w.r.t to the center of the anchor. Since the enhancement is independent of anchor width and height, it can be easily extended to anchor-free object detection approaches like YOLOX, FCOS.

## IoU Based Bounding-box Loss Function

Most modern object detectors optimize advanced variants of IoU loss like GIoU, DIoU or CIoU loss instead of distance-based loss for box detection since these losses are scale-invariant and directly optimizes the evaluation metric itself. Authors use CIoU loss for bounding box supervision. For a ground truth bounding box that is matched with $k^{th}$ anchor at location $(i, j)$ and scale $s$, loss is defined as:

$$ L_{box}(s, i, j, k) = (1 - CIoU(Box_{gt}^{s, i, j, k}, Box_{pred}^{s, i, j, k})) $$

$Box_{pred}^{s, i, j, k}$ is the predicted box for $k^{th}$ anchor at location $(i, j)$ and scale $s$. There are three anchors at each location and prediction happens at four scales.

## Human Pose Loss Function Formulation

OKS is the most popular metric for evaluating keypoints. Conventionally, heat-map based bottom-up approaches use $L_1$ loss to detect keypoints. However, $L_1$ loss may not necessarily be suitable to obtain optimal OKS. Again, $L_1$ loss is naïve and doesn’t take into consideration scale of an object or the type of a keypoint. Since heatmaps are probability maps, it is not possible to use OKS as a loss in pure heatmap based approaches. OKS can be used as a loss function only when keypoints location is regressed. Since, the keypoints are directly regressed w.r.t the anchor center, YOLO-pose can optimize the evaluation metric itself instead of a surrogate loss function. Authors extend the idea of IOU loss from box to keypoints. Object keypoint similarity (OKS) is treated as IOU in case of keypoints. OKS loss is inherently scale-invariant and gives more importance to certain keypoint than others. E.g. Keypoints on a person’s head (eyes, nose, ears) are penalized more for the same pixel-level error than keypoints on a person’s body (shoulders, knees, hips, etc.). These weighting factors are empirically chosen by the COCO authors from redundantly annotated validation images. Unlike vanilla IoU loss, that suffers from vanishing gradient for non-overlapping cases, OKS loss never plateaus. Hence, OKS loss is more similar to DIoU loss.

Corresponding to each bounding box, authors store the entire pose information. Hence, if a ground truth bounding box is matched with anchor at location $(i, j)$ and scale $s$, YOLO-pose predicts the keypoints with respect to the center of the anchor. OKS is computed for each keypoint separately and then summed to give the final OKS loss or keypoint IOU loss.

$$ L_{kpts}(s, i, j, k) = 1 - \sum_{n=1}^{N_{kpts}} OKS = 1 - \frac{\sum_{n=1}^{N_{kpts}} exp (\frac{d_n^2}{2s^2 k_n^2}) \sigma(v_n > 0)}{\sum_{n=1}^{N_{kpts}} \sigma(v_n > 0)}$$

$d_n$ - Euclidean distance between predicted and ground truth location for $n^{th}$ keypoint\
$k_n$ - Keypoint specific weights\
$s$ - Scale of an object\
$\sigma(v_n)$ - visibility flag for each keypoint

Corresponding to each keypoint, a confidence parameter is learned that shows whether a keypoint is present for that person or not. Here, visibility flags for keypoints are used as ground truth

$$ L_{kpts\_conf}(s, i, j, k) = \sum_{n=1}^{N_{kpts}} BCE(\sigma(v_n > 0), p_{kpts}^n) $$

$p_{kpts}^n$ - predicted confidence for $n^{th}$ keypoint

Loss at location $(i, j)$ is valid for $k^{th}$ anchor at scale $s$ if a ground truth bounding box is matched against that anchor. Finally, the total loss is summed over all scales, anchors and locations:

$$ L_{total} = \sum_{s, i, j, k} (λ_{cls} L_{cls} + λ_{box} L_{box} + λ_{kpts} L_{kpts} + λ_{kpts\_conf} L_{kpts\_conf})$$

$λ_{cls} = 0.5$, $λ_{box} = 0.05$, $λ_{kpts} = 0.1$ and $λ_{kpt\_conf} = 0.5$ are hyper-params chosen to balance between losses at different scales.

## Test Time Augmentations

All SoTA approaches for pose estimation rely on Test time augmentations (TTA) to boost performance. Flip test and multi-scale testing are two commonly used techniques. Flip test increases complexity by $2 \times$ whereas Multiscale testing runs inference on three scales {$0.5 \times$, $1 \times$, $2 \times$}, increasing complexity by $(0.25 \times + 1 \times + 4 \times) = 5.25 \times$. With both flip test and multiscale test being on, complexity goes up by $5.25 * 2 \times = 10.5 \times$. Apart from an increase in compute complexity, preparing the augmented data can be expensive in itself. E.g. in flip-test, one needs to flip the image that will add up to the latency of the system. Similarly, multi-scale testing requires resize operation for each scale. These operations can be quite expensive since they may not be accelerated, unlike the CNN operations. Fusing outputs from various forward passes comes with extra cost. For an embedded system, it’s in best interest if one can get competitive result without any TTA. All YOLO-pose results are without any TTA.

## Keypoint Outside Bounding Box

Top-down approaches perform poorly under occlusion. One of the advantages of YOLO-Pose over top-down approaches is there is no constraint for the keypoints to be inside the predicted bounding box. Hence, if keypoints lie outside the bounding box because of occlusion, they can still be correctly recognized. Whereas, in top-down approaches, if the person detection is not correct, pose estimation will fail as well. Both these challenges of occlusion and incorrect box detection are mitigated to some extent in YOLO-pose approach as shown in figure below.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/ef83edaf-a31d-4172-9623-44dd78709568" alt="yolo_pose_examples" height="500"/>
</p>
