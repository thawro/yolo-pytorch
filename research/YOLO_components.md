# Components used in some of the YOLO works

| Year | Done | Approach | Link | Used in | Title |
| :--- | :---: | :---- | :--- | :--- | :---- |
| 2015 | ✅ | [SPP](components/SPP.md) | [paper](https://arxiv.org/pdf/1406.4729.pdf) | v4, v5, PP | Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition |
| 2017 | ✅ | [FPN](components/FPN.md) | [paper](https://arxiv.org/pdf/1612.03144.pdf) | v4 | Feature Pyramid Networks for Object Detection |
| 2017 | ✅ | [Soft NMS](components/Soft-NMS.md) | [paper](https://arxiv.org/pdf/1704.04503.pdf) | v4 | Improving Object Detection With One Line of Code |
| 2017 | ✅ | [Cosine Annealing](components/Cosing-Annealing.md) | [paper](https://arxiv.org/pdf/1608.03983.pdf) | v4 | SGDR: Stochastic Gradient Descent with Warm Restarts |
| 2017 | 🔳 | [SiLU](components/SiLU.md) | [paper](https://arxiv.org/pdf/1702.03118v3) | X | Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning |
| 2017 | ✅ | [Deformable Convolution](components/Deformable-Convolution.md) | [paper](https://arxiv.org/pdf/1703.06211v3) | PP-v1, PP-v2 | Deformable Convolutional Networks |
| 2018 | ✅ | [DropBlock](components/DropBlock.md) | [paper](https://arxiv.org/pdf/1810.12890.pdf) | v4, PP | DropBlock: A regularization method for convolutional networks |
| 2018 | ✅ | [MixUp](components/MixUp.md) | [paper](http://arxiv.org/pdf/1710.09412) | X, PP | MixUp: Beyond Empirical Risk Minimization |
| 2018 | ✅ | [Focal Loss / RetinaNet](components/FocalLoss.md) | [paper](https://arxiv.org/pdf/1708.02002v2.pdf) | v4 | Focal Loss for Dense Object Detection |
| 2018 | ✅ | [CoordConv](components/CoordConv.md) | [paper](https://arxiv.org/pdf/1807.03247v2) | PP | An intriguing failing of convolutional neural networks and the CoordConv solution |
| 2018 | ✅ | [Linear LR Scaling](components/Linear-LR.md) | [paper](https://arxiv.org/pdf/1706.02677) | X, PP-E | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour |
| 2018 | ✅ | [PAN](components/PAN.md) | [paper](https://arxiv.org/pdf/1803.01534v4.pdf) | v6, X, PP | Path Aggregation Network for Instance Segmentation |
| 2018 | ✅ | [GIoU](components/GIoU.md) | [paper](https://arxiv.org/pdf/1902.09630) | PP-E | Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression |
| 2019 | ✅ | [DIoU](components/DIoU.md) | [paper](https://arxiv.org/pdf/1911.08287.pdf) | v4, PP-E | Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression |
| 2019 | ✅ | [Mish](components/Mish.md) | [paper](https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf) | v4, PP-v2 | Mish: A Self Regularized Non-Monotonic Neural Activation Function |
| 2019 | ✅ | [Bag Of Freebies](components/Bag-of-Freebies.md) | [paper](https://arxiv.org/pdf/1902.04103) | X | Bag of Freebies for Training Object Detection Neural Networks |
| 2019 | ✅ | [FCOS](components/FCOS.md) | [paper](https://arxiv.org/pdf/1904.01355) | X, PP-E | FCOS: Fully Convolutional One-Stage Object Detection |
| 2019 | ✅ | [CSP](components/CSP.md) | [paper](https://arxiv.org/pdf/1911.11929.pdf) | X, PP-v2, v6, v9 | CSPNet: A new backbone that can enhance learning capability of CNN |
| 2020 | ✅ | [ATSS](components/ATSS.md) | [paper](https://arxiv.org/pdf/1912.02424) | v6.3 | Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection |
| 2020 | ✅ | [Effective Squeeze and Excitation](components/eSE.md) | [paper](https://arxiv.org/pdf/1911.06667v6) | PP-E | CenterMask : Real-Time Anchor-Free Instance Segmentation |
| 2020 | ✅ | [EfficientDet](components/EfficientDet.md) | [paper](https://arxiv.org/pdf/1911.09070) | v6.3 | EfficientDet: Scalable and Efficient Object Detection |
| 2020 | ✅ | [IoU-Aware](components/IoU-Aware.md) | [paper](https://arxiv.org/pdf/1912.05992) | PP-v2, PP-E | IoU-aware Single-stage Object Detector for Accurate Localization |
| 2020 | ✅ | [Generalized Focal Loss](components/GFL.md) | [paper](https://arxiv.org/pdf/2006.04388) | PP-E | Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection |
| 2020 | ✅ | [Matrix-NMS](components/Matrix-NMS.md) | [paper](https://arxiv.org/pdf/2003.10152v3) | PP-v1, PP-v2 | SOLOv2: Dynamic and Fast Instance Segmentation |
| 2021 | 🔳 | [NoNMS-YOLO](components/NoNMS-YOLO.md) | [paper](https://arxiv.org/pdf/2101.11782) | X | Object Detection Made Simpler by Eliminating Heuristic NMS |
| 2021 | ✅ | [CopyPaste](components/CopyPaste.md) | [paper](https://arxiv.org/pdf/2012.07177) | X | Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation |
| 2021 | ✅ | [OTA](components/OTA.md) | [paper](https://arxiv.org/pdf/2103.14259) | X | OTA: Optimal Transport Assignment for Object Detection |
| 2021 | ✅ | [Varifocal Loss](components/VarifocalLoss.md) | [paper](https://arxiv.org/pdf/2008.13367) | PP-E | VarifocalNet: An IoU-aware Dense Object Detector |
| 2021 | ✅ | [Task Aligned Learning](components/TAL.md) | [paper](https://arxiv.org/pdf/2108.07755) | PP-E | TOOD: Task-aligned One-stage Object Detection |
| 2021 | ✅ | [PP-PicoDet](components/PP-PicoDet.md) | [paper](https://arxiv.org/pdf/2111.00902) | PP-E | PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices |
| 2021 | ✅ | [RepVGG](components/RepVGG.md) | [paper](https://arxiv.org/pdf/2101.03697.pdf) | PP-E, v6 | RepVGG: Making VGG-style ConvNets Great Again |
| 2022 | 🔳 | [ELAN](components/ELAN.md) | [paper](https://arxiv.org/pdf/2211.04800) | v9 | Designing Network Design Strategies Through Gradient Path Analysis |
| 2023 | 🔳 | [PRB-FPN](components/PRB-FPN.md) | [paper](https://arxiv.org/pdf/2012.01724) | v6.3 | Parallel Residual Bi-Fusion Feature Pyramid Network For Accurate Single-Shot Object Detection |
