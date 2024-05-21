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

## Gaussian Modeling

In Gaussian YOLOv3, the bbox coordinates are modeled as Gaussian parameters, consisting of mean ($µ$) and variance ($Σ$). The Gaussian parameters for each bbox coordinate ($t_x$, $t_y$, $t_w$, $t_h$) are processed as follows:
$$ \mu_{t_x} = \sigma(\hat{\mu}{t_x}), \Sigma{t_x} = \sigma(\hat{\Sigma}{t_x})  $$
$$ \mu{t_y} = \sigma(\hat{\mu}{t_y}),  \Sigma{t_y} = \sigma(\hat{\Sigma}{t_y})  $$
$$ \mu{t_w} = \hat{\mu}{t_w},          \Sigma{t_w} = \sigma(\hat{\Sigma}{t_w})  $$
$$ \mu{t_h} = \hat{\mu}{t_h},          \Sigma{t_h} = \sigma(\hat{\Sigma}_{t_h}) $$

where ( $\sigma(x) = \frac{1}{1 + e^{-x}}$ ) is the sigmoid function.

## Reconstruction of Loss Function

The loss function for bbox regression is redesigned as a Negative Log Likelihood (NLL) loss to penalize the model based on the uncertainty of the predicted bbox coordinates:

$$L_x = - \sum_{i=1}^{W} \sum_{j=1}^{H} \sum_{k=1}^{K} \gamma_{ijk} \log(N(x^G_{ijk} | \mu_{t_x}(x_{ijk}), \Sigma_{t_x}(x_{ijk})) + \epsilon)$$

where $L_x$ is the NLL loss of $t_x$ coordinate and the others (i.e., $L_y$ , $L_w$, and $L_h$) are the same as $L_x$ except for each parameter. 

$$\gamma_{ijk} = \frac{\omega_{scale} \times \delta_{obj_{ijk}}}{2}$$
$$\omega_{scale} = 2 - w^G \times h^G$$

## Utilization of Localization Uncertainty

The localization uncertainty, represented as the average of uncertainties of predicted bbox coordinates, is utilized during the detection process. The detection criterion for Gaussian YOLOv3 is defined as:

$$Cr. = \sigma(Object) \times \sigma(Class_i) \times (1 - Uncertainty_{aver})$$

where $\sigma(Object)$ is the objectness score, $\sigma(Class_i)$ is the score of the i-th class, and $Uncertainty_{aver}$ is the average uncertainty of bbox coordinates.

## Benefits

* Improved Accuracy: By modeling bbox coordinates as Gaussian parameters, the algorithm can estimate localization uncertainty, leading to more accurate object detection.
* Robustness to Noisy Data: The redesigned loss function penalizes noisy data during training, making the model more robust and improving performance.
* Reduction in False Positives (FP): Utilizing localization uncertainty helps filter out predictions with high uncertainty, reducing false positives and improving detection precision.
* Increase in True Positives (TP): By considering localization uncertainty during detection, the algorithm can increase true positives, enhancing the overall detection performance and safety of autonomous vehicles.