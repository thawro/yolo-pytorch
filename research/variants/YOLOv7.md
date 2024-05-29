# YOLOv7

2022 | [paper](https://arxiv.org/pdf/2207.02696.pdf) | _YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors_

In this paper, authors present some of the new issues have discovered and devise effective methods to address them. For model reparameterization, they analyze the model re-parameterization strategies applicable to layers in different networks with the concept of gradient propagation path, and propose planned re-parameterized model. In addition, when they discover that with dynamic label assignment technology, the training of model with multiple output layers will generate new issues. That is: _“How to assign dynamic targets for the outputs of different branches?”_ For this problem, authors propose a new label assignment method called **_coarse-to-fine lead guided label assignment_**. The contributions of this paper are summarized as follows

1. Design of several trainable bag-of-freebies methods, so that real-time object detection can greatly improve the detection accuracy without increasing the inference cost
2. For the evolution of object detection methods, authors found two new issues, namely how re-parameterized module replaces original module, and how dynamic label assignment strategy deals with assignment to different output layers. In addition, they also propose methods to address the difficulties arising from these issues
3. Authors propose “extend” and “compound scaling” methods for the real-time object detector that can effectively utilize parameters and computation
4. The proposed method can effectively reduce about 40% parameters and 50% computation of SoTA real-time object detector, and has faster inference speed and higher detection accuracy.

## Extended efficient layer aggregation networks (E-ELAN)

E_ELAN
<p align="center">
  <img src="" alt="" height="450"/>
</p>

Extended-ELAN (E-ELAN) is based on ELAN and its main architecture is shown in above figure (d). Regardless of the gradient path length and the stacking number of computational blocks in large-scale ELAN, it has reached a stable state. If more computational blocks are stacked unlimitedly, this stable state may be destroyed, and the parameter utilization rate will decrease. The proposed E-ELAN uses expand, shuffle and merge cardinality to achieve the ability to continuously enhance the learning ability of the network without destroying the original gradient path. In terms of architecture, E-ELAN only changes the architecture in computational block, while the architecture of transition layer is completely unchanged. The proposed strategy is to use group convolution to expand the channel and cardinality of computational blocks. Authors apply the same group parameter and channel multiplier to all the computational blocks of a computational layer. Then, the feature map calculated by each computational block will be shuffled into $g$ groups according to the set group parameter $g$, and then concatenate them together. At this time, the number of channels in each group of feature map will be the same as the number of channels in the original architecture. Finally, we add $g$ groups of feature maps to perform merge cardinality. In addition to maintaining the original ELAN design architecture, E-ELAN can also guide different groups of computational blocks to learn more diverse features.

## Model scaling for concatenation-based models

yolo_v7_scaling
<p align="center">
  <img src="" alt="" height="300"/>
</p>

The main purpose of model scaling is to adjust some attributes of the model and generate models of different scales to meet the needs of different inference speeds. For example the scaling model of EfficientNet considers the width, depth, and resolution. As for the scaled-YOLOv4, its scaling model is to adjust the number of stages. In other works, authors analyzed the influence of vanilla convolution and group convolution on the amount of parameter and computation when performing width and depth scaling, and used this to design the corresponding model scaling method. The above methods are mainly used in architectures such as PlainNet or ResNet. When these architectures are in executing scaling up or scaling down, the in-degree and out-degree of each layer will not change, so we can independently analyze the impact of each scaling factor on the amount of parameters and computation. However, if these methods are applied to the concatenation-based architecture, we will find that when scaling up or scaling down is performed on depth, the in-degree of a translation layer which is immediately after a concatenation-based computational block will decrease or increase, as shown in figure above (a) and (b). It can be inferred from the above phenomenon that we cannot analyze different scaling factors separately for a concatenation-based model but must be considered together. Take scaling-up depth as an example, such an action will cause a ratio change between the input channel and output channel of a transition layer, which may lead to a decrease in the hardware usage of the model. Therefore, authors proposed the corresponding compound model scaling method for a concatenation-based model. When scaling the depth factor of a computational block, we must also calculate the change of the output channel of that block. Then, we will perform width factor scaling with the same amount of change on the transition layers. The result is shown in figure above (c). The proposed compound scaling method can maintain the properties that the model had at the initial design and maintains the optimal structure.

## Trainable bag-of-freebies

### Planned re-parameterized convolution

yolov7_reparametrized
<p align="center">
  <img src="" alt="" height="500"/>
</p>

Although RepConv (from RepVGG) has achieved excellent performance on the VGG, when it is directly applied to ResNet, DenseNet and other architectures, its accuracy will be significantly reduced. Authors use gradient flow propagation paths to analyze how re-parameterized convolution should be combined with different network. They also designed planned re-parameterized convolution accordingly.

RepConv actually combines $3 × 3$ convolution, $1 × 1$ convolution, and identity connection in one convolutional layer. After analyzing the combination and corresponding performance of RepConv and different architectures, authors find that the identity connection in RepConv destroys the residual in ResNet and the concatenation in DenseNet, which provides more diversity of gradients for different feature maps. For the above reasons, they use RepConv without identity connection (RepConvN) to design the architecture of planned re-parameterized convolution. They motivate it saying that, when a convolutional layer with residual or concatenation is replaced by re-parameterized convolution, there should be no identity connection. Figure above shows an example of the designed “planned re-parameterized convolution” used in PlainNet and ResNet.

### Coarse for auxiliary and fine for lead loss

yolov7_head
<p align="center">
  <img src="" alt="" height="300"/>
</p>

Deep supervision is a technique that is often used in training deep networks. Its main concept is to add extra auxiliary head in the middle layers of the network, and the shallow network weights with assistant loss as the guide. Even for architectures such as ResNet and DenseNet which usually converge well, deep supervision can still significantly improve the performance of the model on many tasks. Figure above (a) and (b) show, respectively, the object detector architecture “without” and “with” deep supervision. In this paper, authors call the head responsible for the final output as the **_lead head_**, and the head used to assist training is called **_auxiliary head_**.

**Label assignment**. In the past, in the training of deep network, label assignment usually refers directly to the ground truth and generate hard label according to the given rules. However, in recent years, if we take object detection as an example, researchers often use the quality and distribution of prediction output by the network, and then consider together with the ground truth to use some calculation and optimization methods to generate a reliable soft label. For example, YOLO use IoU of prediction of bounding box regression and ground truth as the soft label of objectness. In this paper, authors call the mechanism that considers the network prediction results together with the ground truth and then assigns soft labels as “**_label assigner_**.”

Deep supervision needs to be trained on the target objectives regardless of the circumstances of auxiliary head or lead head. During the development of soft label assigner related techniques, authors discovered a new derivative issue, i.e., “_How to assign soft label to auxiliary head and lead head ?_”. The results of the most popular method at present is as shown in figure above (c), which is to separate auxiliary head and lead head, and then use their own prediction results and the ground truth to execute label assignment. The method proposed in this paper is a new label assignment method that guides both auxiliary head and lead head by the lead head prediction. In other words, authors use lead head prediction as guidance to generate _coarse-to-fine_ hierarchical labels, which are used for auxiliary head and lead head learning, respectively. The two proposed deep supervision label assignment strategies are shown in above figure (d) and (e), respectively.

**Lead head guided label assigner** is mainly calculated based on the prediction result of the lead head and the ground truth, and generate soft label through the optimization process. This set of soft labels will be used as the target training model for both auxiliary head and lead head. The reason to do this is because lead head has a relatively strong learning capability, so the soft label generated from it should be more representative of the distribution and correlation between the source data and the target. Furthermore, we can view such learning as a kind of generalized residual learning. By letting the shallower auxiliary head directly learn the information that lead head has learned, lead head will be more able to focus on learning residual information that has not yet been learned.

**Coarse-to-fine lead head guided label assigner** also uses the predicted result of the lead head and the ground truth to generate soft label. However, in the process we generate two different sets of soft label, i.e., _coarse label_ and _fine label_, where fine label is the same as the soft label gen- erated by lead head guided label assigner, and coarse label is generated by allowing more grids to be treated as positive target by relaxing the constraints of the positive sample assignment process. The reason for this is that the learning ability of an auxiliary head is not as strong as that of a lead head, and in order to avoid losing the information that needs to be learned, the focus is put on optimizing the recall of auxiliary head in the object detection task. As for the output of lead head, it is possible to filter the high precision results from the high recall results as the final output. However, if the additional weight of coarse label is close to that of fine label, it may produce bad prior at final prediction. Therefore, in order to make those extra coarse positive grids have less impact, a restrictions are put in the decoder, so that the extra coarse positive grids cannot produce soft label perfectly. The mechanism mentioned above allows the importance of fine label and coarse label to be dynamically adjusted during the learning process, and makes the optimizable upper bound of fine label always higher than coarse label.

### Other trainable bag-of-freebies

These freebies are some of the tricks authors used in training, but the original concepts were not proposed by them:

1. **Batch normalization in conv-bn-activation topology** - This part mainly connects batch normalization layer directly to convolutional layer. The purpose of this is to integrate the mean and variance of batch normalization into the bias and weight of convolutional layer at the inference stage
2. **Implicit knowledge in YOLOR combined with convolution feature map** in addition and multiplication manner - Implicit knowledge in YOLOR can be simplified to a vector by pre-computing at the inference stage. This vector can be combined with the bias and weight of the previous or subsequent convolutional layer
3. **EMA model** - EMA is a technique used in mean teacher, and in YOLOv7 system authors use EMA model purely as the final inference model.
