# YOLOX

2021 | [paper](https://arxiv.org/pdf/2107.08430.pdf) | _YOLOX: Exceeding YOLO Series in 2021_

In this report, authors present some experienced improvements to YOLO series, forming a new high-performance detector — YOLOX. They switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve SoTA results across a large scale range of models.

## YOLOX-DarkNet53

YOLOv3 with Darknet53 is chosen as baseline. The following part, walks through the whole system designs in YOLOX step by step.

### Implementation details

The training settings are mostly consistent from the baseline to our final model:

* Model is trained for a total of $300$ epochs with $5$ epochs warmup on COCO train2017
* **_Stochastic Gradient Descent (SGD)_** is applied for training with a learning rate of $\frac{lr * BatchSize}{64}$ (**_linear scaling_**), with an initial $lr = 0.01$ and the **_cosine lr schedule_**
* The weight decay is $0.0005$ and the SGD momentum is $0.9$
* The batch size is $128$ by default to typical 8-GPU devices (other batch sizes for single GPU training also work well)
* The input size is evenly drawn from $448$ to $832$ with $32$ strides

FPS and latency in this report are all measured with FP16-precision and batch=1 on a single Tesla V100.

### YOLOv3 baseline

The baseline adopts the architecture of DarkNet53 backbone and an SPP layer, referred to **YOLOv3-SPP** in some papers. Authors slightly change some training strategies compared to the original implementation, adding **_EMA_** weights updating, **_cosine lr schedule_**, **_IoU loss_** and **_IoU-aware_** branch. We use BCE Loss for training _cls_ and _obj_ branch, and IoU Loss for training _reg_ branch. These general training tricks are orthogonal to the key improvement of YOLOX, authors thus put them on the baseline. Moreover, they only conduct _RandomHorizontalFlip_, _ColorJitter_ and _multi-scale_ for data augmentation and discard the _RandomResizedCrop_ strategy, because they found the _RandomResizedCrop_ is kind of overlapped with the planned _Mosaic_ augmentation. With those enhancements, the baseline achieves 38.5% AP on COCO val.

### Decoupled head

yolox_head

yolox_head_converge
yolox_head_metrics

In object detection, the conflict between classification and regression tasks is a well-known problem. Thus the decoupled head for classification and localization is widely used in the most of one-stage and two-stage detectors. However, as YOLO series’ backbones and feature pyramids (e.g., FPN, PAN) continuously evolve, their detection heads remain coupled as shown in the above figure. Two analytical experiments indicate that the coupled detection head may harm the performance.

1. Replacing YOLO’s head with a decoupled one greatly improves the converging speed as shown in figure above
2. The decoupled head is essential to the end-to-end version of YOLO

One can tell from the table above, that the end-to-end property decreases by 4.2% AP with the coupled head, while the decreasing reduces to 0.8% AP for a decoupled head. Authors thus replace the YOLO detect head with a lite decoupled head as shown in figure above. Concretely, it contains a $1 × 1$ _conv_ layer to reduce the channel dimension, followed by two parallel branches with two $3 × 3$ _conv_ layers respectively. The lite decoupled head brings additional 1.1 ms inference time (11.6 ms v.s. 10.5 ms).

### Strong data augmentation

YOLOX adds _Mosaic_ and _MixUp_ into the augmentation strategies to boost YOLOX’s performance. _Mosaic_ is an efficient augmentation strategy proposed by ultralytics-YOLOv3. It is then widely used in YOLOv4, YOLOv5 and other detectors. _MixUp_ is originally designed for image classification task but then modified in Bag of Freebies paper for object detection training. Authors adopt the _MixUp_ and _Mosaic_ implementation in the model and close it for the last 15 epochs, achieving 42.0% AP. After using strong data augmentation, they found ImageNet pre-training is no more beneficial, **thus they train all the following models from scratch**.

### Anchor-free

Both YOLOv4 and YOLOv5 follow the original anchor-based pipeline of YOLOv3. However, the anchor mechanism has many known problems:

1. To achieve optimal detection performance, one needs to conduct clustering analysis to determine a set of optimal anchors before training. Those clustered anchors are domain-specific and less generalized
2. Anchor mechanism increases the complexity of detection heads, as well as the number of predictions for each image. On some edge AI systems, moving such large amount of predictions between devices (e.g., from NPU to CPU) may become a potential bottleneck in terms of the overall latency.

Anchor-free detectors have developed rapidly in the past two years. These works have shown that the performance of anchor-free detectors can be on par with anchor-based detectors. Anchor-free mechanism significantly reduces the number of design parameters which need heuristic tuning and many tricks involved (e.g., Anchor Clustering, Grid Sensitive) for good performance, making the detector, especially its training and decoding phase, considerably simpler. Switching YOLO to an anchor-free manner is quite simple. YOLOX reduces the predictions for each location from 3 to 1 and make them directly predict four values, i.e., two offsets in terms of the left-top corner of the grid, and the height and width of the predicted box. Authors assign the center location of each object as the positive sample and pre-define a scale range, as done in FCOS, to designate the FPN level for each object. Such modification reduces the parameters and GFLOPs of the detector (makes it faster) and obtains better performance – 42.9% AP.

### Multi positives

To be consistent with the assigning rule of YOLOv3, the above anchor-free version selects only ONE positive sample (the center location) for each object meanwhile ignores other high quality predictions. However, optimizing those high quality predictions may also bring beneficial gradients, which may alleviate the extreme imbalance of positive/negative sampling during training. YOLOX simply assigns the center $3 × 3$ area as positives, also named “center sampling” in FCOS. The performance of the detector improves to 45.0% AP.

### SimOTA

Advanced label assignment is another important progress of object detection in recent years. Based on authors previous study (OTA), they conclude four key insights for an advanced label assignment:

1. Loss/quality aware
2. Center prior
3. Dynamic number of positive anchors for each ground-truth (abbreviated as dynamic _top-k_)
4. Global view

OTA meets all four rules above, hence YOLOX chooses it as a candidate label assigning strategy. Specifically, OTA analyzes the label assignment from a global perspective and formulates the assigning procedure as an _Optimal Transport (OT)_ problem, producing the SoTA performance among the current assigning strategies. However, in practice solving OT problem via _Sinkhorn-Knopp_ algorithm brings 25% extra training time, which is quite expensive for training 300 epochs. Authors thus simplify it to dynamic _top-k_ strategy, named **_SimOTA_**, to get an approximate solution.

SimOTA first calculates pair-wise matching degree, represented by cost or quality for each prediction-gt pair. For example, in SimOTA, the cost between gt $g_i$ and prediction $p_j$ is calculated as:

$$ c_{ij} = L_{ij}^{cls} + λ L_{ij}^{reg} $$

where $λ$ is a balancing coefficient. $L_{ij}^{cls}$ and $L_{ij}^{reg}$ are classficiation loss and regression loss between gt $g_i$ and prediction $p_j$. Then, for gt $g_i$, select the top $k$ predictions with the least cost within a fixed center region as its positive samples. Finally, the corresponding grids of those positive predictions are assigned as positives, while the rest grids are negatives. Noted that the value $k$ varies for different ground-truth. Please refer to Dynamic $k$ Estimation strategy in OTA for more details. SimOTA not only reduces the training time but also avoids additional solver hyperparameters in _Sinkhorn-Knopp_ algorithm. SimOTA raises the detector from 45.0% AP to 47.3% AP, showing the power of the advanced assigning strategy.

### End-to-end YOLO 

We follow [paper](https://arxiv.org/pdf/2101.11782) to add two additional _conv_ layers, one-to-one label assignment, and stop gradient. These enable the detector to perform an end-to-end manner, but slightly decreasing the performance and the inference speed. Authors thus leave it as an optional module which is not involved in the final models.

## Other Backbones

Besides DarkNet53, authors also tested YOLOX on other backbones with different sizes, where YOLOX achieves consistent improvements against all the corresponding counterparts.

### Modified CSPNet in YOLOv5

To give a fair comparison, authors adopts the exact YOLOv5’s backbone including modified CSPNet, SiLU activation, and the PAN head. They also follow its scaling rule to product _YOLOX-S_, _YOLOX-M_, _YOLOX-L_, and _YOLOX-X_ models. Compared to YOLOv5, YOLOX models get consistent improvement by ∼3.0% to ∼1.0% AP, with only marginal time increasing (comes from the decoupled head).

### Tiny and Nano detectors

Authors further shrink the model to _YOLOX-Tiny_ to compare with YOLOv4-Tiny. For mobile devices, YOLOX adopts depth wise convolution to construct a _YOLOX-Nano_ model, which has only 0.91M parameters and 1.08G FLOPs. YOLOX performs well with even smaller model size than the counterparts.

### Model size and data augmentation

In the experiments, all the models keep almost the same learning schedule and optimizing parameters as depicted earlier. However, authors have found that the suitable augmentation strategy varies across different size of models. While applying MixUp for YOLOX-L can improve AP by 0.9%, it is better to weaken the augmentation for small models like YOLOX-Nano. Specifically, authors remove the _Mixup_ augmentation and weaken the _Mosaic_ (reduce the scale range from $[0.1, 2.0]$ to $[0.5, 1.5]$) when training small models, i.e., YOLOX-S, YOLOX-Tiny, and YOLOX-Nano. Such a modification improves YOLOX-Nano’s AP from 24.0% to 25.3%. For large models, authors also found that stronger augmentation is more helpful. Indeed, YOLOX's _MixUp_ implementation is part of heavier than the original version. Inspired by Copypaste, authors jittered both images by a random sampled scale factor before mixing them up. To understand the power of _Mixup_ with scale jittering, authors compare it with _Copypaste_ on YOLOX-L. Noted that _Copypaste_ requires extra instance mask annotations while _Mixup_ does not. As shown in experiments, these two methods achieve competitive performance, indicating that _Mixup_ with scale jittering is a qualified replacement for _Copypaste_ when no instance mask annotation is available.
