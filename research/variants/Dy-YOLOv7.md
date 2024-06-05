# Dy-YOLOv7

2023 | [paper](https://arxiv.org/pdf/2304.05552) | _DynamicDet: A Unified Dynamic Architecture for Object Detection_

Dynamic neural network is an emerging research topic in deep learning. With adaptive inference, dynamic models can achieve remarkable accuracy and computational efficiency. However, it is challenging to design a powerful dynamic detector, because of no suitable dynamic architecture and exiting criterion for object detection. To tackle these difficulties, authors propose a dynamic framework for object detection, named DynamicDet. Firstly, they carefully design a dynamic architecture based on the nature of the object detection task. Then propose an adaptive router to analyze the multi-scale information and to decide the inference route automatically. Authors also present a novel optimization strategy with an exiting criterion based on the detection losses for the dynamic detectors. Last, they present a variable-speed inference strategy, which helps to realize a wide range of accuracy-speed trade-offs with only one dynamic detector.

The human brain inspires many fields of deep learning, and the dynamic neural network is a typical one. The processing speeds of images are different in our brains, which depend on the difficulties of the images. This property motivates the image-wise dynamic neural network, and many exciting works have been proposed (e.g., Branchynet, MSDNet, DVT). Although these approaches have achieved remarkable performance, they are all designed specifically for the image classifica- tion task and are not suitable for other vision tasks, especially for the object detection. The main difficulties in designing an image-wise dynamic detector are as follows:

**Dynamic detectors cannot utilize the existing dynamic architectures**. Most existing dynamic architectures are cascaded with multiple stages (i.e., a stack of multiple layers), and predict whether to stop the inference at each exiting point. Such a paradigm is feasible in image classification but is ineffective in object detection, since an image has multiple objects and each object usually has different categories and scales. Hence, almost all detectors depend heavily on multi-scale information, utilizing the features on different scales to detect objects of different sizes (which are obtained by fusing the multi-scale features of the backbone with a detection neck, i.e., FPN). In this case, the exiting points for detectors can only be placed behind the last stage. Consequently, the entire backbone module has to be run completely, and it is impossible to achieve dynamic inference on multiple cascaded stages.

**Dynamic detectors cannot exploit the existing exiting criteria for image classification**. For the image classification task, the threshold of top-1 accuracy is a widely used criterion for decision-making. Notably, it only needs one fully connected layer to predict the top-1 accuracy at intermediate layer, which is easy and costless. However, object detection task requires the neck and the head to predict the categories and locations of the object instances. Hence, the existing exiting criteria for image classification is not suitable for object detection

To deal with the above difficulties, authors propose a dynamic framework to achieve dynamic inference for object detection, named **_DynamicDet_**. Firstly, they design a dynamic architecture for the object detection task, which can exit with multi-scale information during the inference. Then, they propose an adaptive router to choose the best route for each image automatically. Besides, they present the corresponding optimization and inference strategies for the proposed DynamicDet. The main contributions are as follows:

* Propose a dynamic architecture for object detection, named **_DynamicDet_**, which consists of two cascaded detectors and a router. This dynamic architecture can be easily adapted to mainstream detectors, e.g., Faster R-CNN and YOLO.
* Propose an adaptive router to predict the difficulty scores of the images based on the multi-scale features, and achieve automatic decision-making. In addition, authors propose a hyperparameter-free optimization strategy and a variable-speed inference strategy for our dynamic architecture.

dy_yolo

<p align="center">
  <img src="" alt="" height="300"/>
</p>

## Overall architecture

The overall architecture of the dynamic detector is shown in figure above. Inspired by CBNet, the dynamic architecture consists of two detectors and one router. For an input image $x$, authors initially extract its multi-scale features $F_1$ with the first backbone $B_1$ as:

$$ F_1 = B_1(x) = [f_1^1, f_1^2, ..., f_1^L] $$

where $L$ denotes the number of stages, i.e., the number of multi-scale features. Then, the router $R$ will be fed with these features $F_1$ to predict a difficulty score $ϕ ∈ (0, 1)$ for this image as:

$$ ϕ = R(F_1) $$

Generally speaking, the "easy" images exit at the first backbone, while the "hard" images require the further processing. Specifically, if the router classifies the input image as an "easy" one, the following neck and head $D_1$ will output the detection results $y$ as:

$$ y = D_1(F_1) $$

On the contrary, if the router classifies the input image as a "hard" one, the multi-scale features will need further enhancement by the second backbone, instead of immediately decoded by $D_1$. In particular, authors embed the multi-scale features $F_1$ into $H$ by a composite connection module $G$ as:

$$ H = G(F_1) = [h^1, h^2, ..., h^L] $$

where $G$ is the DHLC of CBNet. Then, authors feed the input image $x$ into the second backbone and enhance the features of the second backbone via summing the corresponding elements of $H$ at each stage sequentially, denoted as:

$$ F_2 = B_2(x, H) = [f_2^1, f_2^2, ..., f_2^L] $$

and the detection results will be obtained by the second head and neck $D_2$ as:

$$ y = D_2(F_2) $$

Through the above process, the "easy" images will be processed by only one backbone, while the "hard" images will be processed by two. Obviously, with such an architecture, trades-offs between computation (i.e., speed) and accuracy can be achieved.

## Adaptive router

In mainstream object detectors, different scale features play different roles. Generally, the features of the shallow layers, with strong spatial information and small receptive fields, are more used to detect small objects. In contrast, the features of the deep layers, with strong semantic information and large receptive fields, are more used to detect large objects. This property makes it necessary to consider multi-scale information when predicting the difficulty score of an image. According to this, authors design an adaptive router based on the multi-scale features, that is, a simple yet effective decision-maker for the dynamic detector.

Inspired by the squeeze-and-excitation (SE) module, authors first pool the multi-scale features $F_1$ independently and concatenate them all as:

$$ \tilde{F_1} = C([P(f_1^1), P(f_1^2), ..., P(f_1^L)]) $$

where $P$ denotes the global average pooling and $C$ denotes the channel-wise concatenation. With this operation, authors compress the multi-scale features $F_1$ into a vector $\tilde{F_1} ∈ R^d$ of dimension $d$. Then, they map this vector to a difficulty score $ϕ ∈ (0, 1)$ via two learnable fully connected layers as:

$$ ϕ = σ(W_2(δ(W_1 \tilde{F_1} + b_1)) + b_2) $$

where $δ$, $σ$ denote the _ReLU_ and _Sigmoid_ activation functions respectively, and $W_1$, $W_2$, $b_1$, $b_2$ are learnable parameters. Following other work, authors reduce the feature dimension to $⌊d/4⌋$ in the first fully connected layer, and exploit the second fully connected layer with a _Sigmoid_ function to generate the predicted score. It is worth noting that the computational burden of this router can be negligible since multi-scale features are first polled to one vector.

## Optimization strategy

Firstly, authors jointly train the cascaded detectors, and the training objective is:

$$ \min_{Θ1,Θ2} (L_{det}^1 (x, y | Θ_1) + L_{det}^2 (x, y | Θ_2)) $$

where $x$, $y$ denote the input image and the ground truth respectively, $Θ_i$ denotes the learnable parameters of the detector $i$ and $L_{det}^i$ denotes the training loss for detector $i$ (e.g., bounding box regression loss and classification loss). After the above training phase, these two detectors will be able to detect the objects, and authors freeze their parameters $Θ_1$, $Θ_2$ during the later training.

Then, they train the adaptive router to automatically distinguish the difficulty of the image. Here, authors assume the parameters of the router are $Θ_R$ and the predicted difficulty score obtained is $ϕ$. We hope the router can assign the "easy" images (i.e., with lower $ϕ$) to the faster detector (i.e., the first detector) and the "hard" images (i.e., with higher $ϕ$) to the more accurate detector (i.e., the second detector).

However, it is non-trivial to implement that in practice. If we directly optimize the router without any constraints as:

$$ \min_{Θ_R} ((1 − ϕ)L_{det}^1 (x, y | Θ_1) + ϕL_{det}^2 (x, y | Θ_2)) $$

the router will always choose the most accurate detector as it allows for a lower training loss. Furthermore, if we naively add hardware constraints to the training objective as:

$$ \min_{Θ_R} ((1 − ϕ)L_{det}^1 (x, y | Θ_1) + ϕL_{det}^2 (x, y | Θ_2) +λϕ ) $$

we will have to adjust the hyperparameter $λ$ by try and error, leading to huge workforce consumption. To overcome the above challenges, authors propose a hyperparameter-free optimization strategy for the adaptive router. First, they define the difficulty criterion based on the corresponding training loss difference between two detectors of an image. Specifically, they assume that if the loss difference of an image between two detectors is small enough, this image can be classified as an "easy" image. Instead, if the loss difference is large enough, it should be classified as a "hard" image. Ideally, for a balanced situation, authors hope the easier half of all images go through the first detector, and the harder half go through the second one. To achieve this, they introduce an adaptive offset to balance the losses of two detectors and optimize our router via gradient descent. In practice, authors first calculate the median of the training loss difference $∆$ between the first and the second detector on the training set. Then, the training objective of our router can be formulated as:

$$ \min_{Θ_R} ((1 − ϕ)(L_{det}^1 (x, y | Θ_1) - \frac{∆}{2}) + ϕ(L_{det}^2 (x, y | Θ_2) + \frac{∆}{2})) $$

where $\frac{∆}{2}$ is used to reward the first detector and punish the second detector, respectively. Without this reward and penalty, the losses of the second detector are always smaller than the first detector. When the reward and penalty are conducted, their loss curves intersect and reveal the optimal curve. The training objective provides a means to optimize the adaptive router by introducing the following gradient through the difficulty score $ϕ$ to all parameters $Θ_R$ of the router as:

dy_yolo_grad

<p align="center">
  <img src="" alt="" height="80"/>
</p>

To distinguish between "easy" and "hard" images better, authors expect the optimization direction of the router to be related to the difficulty of the image, i.e., the difference in loss between the two detectors. Obviously, the gradient at above equation enable such expectation.

## Variable-speed inference

Authors further propose a simple and effective method to determine the difficulty score thresholds to achieve variable-speed inference with only one dynamic detector. Specifically, the adaptive router will output a difficulty score and decide which detector to go through based on a certain threshold during inference. Therefore, one can set different thresholds to achieve different accuracy-speed trade-offs. Firstly, authors count the difficulty scores $S_{val}$ of the valida- tion set. Then, based on the actual needs (e.g., the target latency), one can obtain the corresponding threshold for the router. For example, assuming the latency of the first detector is $lat_1$, the latency of the cascaded two detectors is $lat_2$ and the target latency is $lat_t$, one can calculate the maximum allowable proportion of the "hard" images $k$ as:

$$ k = \frac{lat_t − lat_1}{lat_2 − lat_1} $$

$$ lat_1 ≤ lat_t ≤ lat_2 $$

and then the threshold $τ_{val}$ will be:

$$ τ_{val} = percentile(S_{val}, k) $$

where $percentile(·, k)$ means to compute the $k$-th quantile of the data. It is worth noting that this threshold $τ_{val}$ is robust in both validation set and test set because these two sets are independent and identically distributed (i.i.d.).

Based on the above strategy, one dynamic detector can directly cover the accuracy-speed trade-offs from the single to double detectors, avoiding redesigning and training multiple detectors under different hardware constraints.
