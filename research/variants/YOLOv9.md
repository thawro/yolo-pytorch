# YOLOv9

2023 | [paper](https://arxiv.org/pdf/2402.13616) | _YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information_

Today’s deep learning methods focus on how to design the most appropriate objective functions so that the prediction results of the model can be closest to the ground truth. Meanwhile, an appropriate architecture that can facilitate acquisition of enough information for prediction has to be designed. Existing methods ignore a fact that when input data undergoes layer-by-layer feature extraction and spatial transformation, large amount of information will be lost. This paper delves into the important issues of data loss when data is transmitted through deep networks, namely **_information bottleneck_** and **_reversible functions_**.

Authors proposed the concept of **_Programmable Gradient Information (PGI)_** to cope with the various changes required by deep networks to achieve multiple objectives. PGI can provide complete input information for the target task to calculate objective function, so that reliable gradient information can be obtained to update network weights.\
In addition, a new lightweight network architecture – **_Generalized Efficient Layer Aggregation Network (GELAN)_**, based on gradient path planning is designed. GELAN’s architecture confirms that PGI has gained superior results on lightweight models. Authors verified the proposed GELAN and PGI on MS COCO dataset based object detection. The results show that GELAN (which uses only conventional convolution operators) achieves better parameter utilization than the SoTA methods developed based on _depth-wise convolution_. PGI can be used for variety of models from lightweight to large. It can be used to obtain complete information, so that train-from-scratch models can achieve better results than SoTA models pre-trained using large datasets.

<p align="center">
  <img src="" alt="yolo_v9_gelan_comparison" height="300"/>
</p>

In recent years, researchers in the field of deep learning have mainly focused on how to develop more powerful system architectures and learning methods, such as _CNNs_, _Transformers_, _Perceivers_, and _Mambas_. In addition, some researchers have tried to develop more general objective functions, such as _loss function_, _label assignment_ and _auxiliary supervision_. The above studies all try to precisely find the mapping between input and target tasks. However, most past approaches have ignored that input data may have a non-negligible amount of information loss during the feedforward process. This loss of information can lead to biased gradient flows, which are subsequently used to update the model. The above problems can result in deep networks to establish incorrect associa- tions between targets and inputs, causing the trained model to produce incorrect predictions.

In deep networks, the phenomenon of input data losing information during the feedforward process is commonly known as **_information bottleneck_**, and its schematic diagram is as shown in figure above. At present, the main methods that can alleviate this phenomenon are as follows:

1. **Reversible architectures** - this method mainly uses repeated input data and maintains the information of the input data in an explicit way
2. **Masked modeling** - it mainly uses reconstruction loss and adopts an implicit way to maximize the extracted features and retain the input information
3. **Deep supervision** - it uses shallow features that have not lost too much important information to pre-establish a mapping from features to targets to ensure that important information can be transferred to deeper layers

However, the above methods have different drawbacks in the training process and inference process. To address the above-mentioned issues, authors propose a new concept, which is **_Programmable Gradient Information (PGI)_**. The concept is to generate reliable gradients through auxiliary reversible branch, so that the deep features can still maintain key characteristics for executing target task. The design of auxiliary reversible branch can avoid the semantic loss that may be caused by a traditional deep supervision process that integrates multi-path features. In other words, authors are **programming gradient information propagation at different semantic levels**, and thereby achieving the best training results. The reversible architecture of PGI is built on auxiliary branch, so there is no additional cost. Since PGI can freely select loss function suitable for the target task, it also overcomes the problems encountered by mask modeling. The proposed PGI mechanism can be applied to deep neural networks of various sizes and is more general than the deep supervision mechanism, which is only suitable for very deep neural networks.

In this paper, authors also designed **Generalized0-ELAN (GELAN)** based on ELAN (from YOLOv7), the design of GELAN simultaneously takes into account the number of parameters, computational complexity, accuracy and inference speed. This design allows users to **arbitrarily choose appropriate computational blocks** for different inference devices. Authors have combined the proposed PGI and GELAN, and then designed a new generation of YOLO series object detection system, called **_YOLOv9_**.

The main contributions of this paper as follows:

1. Theoretically analyzed the existing deep neural network architecture from the perspective of reversible function, and through this process successfully ex- plained many phenomena that were difficult to explain in the past. Authors also designed PGI and auxiliary reversible branch based on this analysis and achieved excellent results
2. The designed PGI solves the problem that deep supervision can only be used for extremely deep neural network architectures, and therefore allows new lightweight architectures to be truly applied in daily life
3. The designed GELAN only uses conventional _convolution_ to achieve a higher parameter usage than the _depth-wise convolution_ design that based on the most advanced technology, while showing great advantages of being light, fast, and accurate.

## Core innovations

YOLOv9's advancements are deeply rooted in addressing the challenges posed by information loss in deep neural networks. The **Information Bottleneck Principle** and the innovative use of **Reversible Functions** are central to its design, ensuring YOLOv9 maintains high efficiency and accuracy.

### Information Bottleneck Principle

The Information Bottleneck Principle reveals a fundamental challenge in deep learning: as data passes through successive layers of a network, the potential for information loss increases. This phenomenon is mathematically represented as:

$$ I(X, X) >= I(X, f_{\theta}(X)) >= I(X, g_{\phi}(f_{\theta}(X))) $$

where $I$ denotes mutual information, and $f$ and $g$ represent transformation functions with parameters $\theta$ and $\phi$, respectively. YOLOv9 counters this challenge by implementing **_Programmable Gradient Information (PGI)_**, which aids in preserving essential data across the network's depth, ensuring more reliable gradient generation and, consequently, better model convergence and performance.

### Reversible Functions

The concept of Reversible Functions is another cornerstone of YOLOv9's design. A function is deemed reversible if it can be inverted without any loss of information, as expressed by:

$$ X = v_{\zeta}(r_{\psi}(X)) $$

with $\psi$ and $\zeta$ as parameters for the reversible and its inverse function, respectively. This property is crucial for deep learning architectures, as it allows the network to retain a complete information flow, thereby enabling more accurate updates to the model's parameters. YOLOv9 incorporates reversible functions within its architecture to mitigate the risk of information degradation, especially in deeper layers, ensuring the preservation of critical data for object detection tasks.

### Impact on Lightweight Models

Addressing information loss is particularly vital for lightweight models, which are often under-parameterized and prone to losing significant information during the feedforward process. YOLOv9's architecture, through the use of PGI and reversible functions, ensures that even with a streamlined model, the essential information required for accurate object detection is retained and effectively utilized.

## Programmable Gradient Information

<p align="center">
  <img src="" alt="yolo_v9_PIG" height="500"/>
</p>

In order to solve the aforementioned problems, authors propose a new auxiliary supervision framework called **_Programmable Gradient Information (PGI)_**, as shown in figure above (d). PGI mainly includes three components, namely:

1. **Main branch**
2. **Auxiliary reversible branch** - designed to deal with the problems caused by the deepening of neural networks. Network deepening will cause information bottleneck, which will make the loss function unable to generate reliable gradients.
3. **Multi-level auxiliary information** - designed to handle the error accumulation problem caused by deep supervision

From figure above (d) one can see that the inference process of PGI only uses main branch and therefore does not require any additional inference cost. As for the other two components, they are used to solve or slow down several important issues in deep learning methods.

### Auxiliary Reversible Branch

In PGI, authors propose auxiliary reversible branch to generate reliable gradients and update network parameters. By providing information that maps from data to targets, the loss function can provide guidance and avoid the possibility of finding false correlations from incomplete feedforward features that are less relevant to the target. Authors propose the maintenance of complete information by introducing reversible architecture, but adding main branch to reversible architecture will consume a lot of inference costs. They analyzed the architecture of figure above (b) and found that when additional connections from deep to shallow layers are added, the inference time will increase by 20%. When they repeatedly add the input data to the high-resolution computing layer of the network (yellow box), the inference time even exceeds twice the time.

Since the goal is to use reversible architecture to obtain reliable gradients, “reversible” is not the only necessary condition in the inference stage. In view of this, authors regard reversible branch as an expansion of deep supervision branch, and then design auxiliary reversible branch, as shown in figure above (d). As for the main branch deep features that would have lost important information due to information bottleneck, they will be able to receive reliable gradient information from the auxiliary reversible branch. These gradient information will drive parameter learning to assist in extracting correct and important information, and the above actions can enable the main branch to obtain features that are more effective for the target task. Moreover, the reversible architecture performs worse on shallow networks than on general networks because complex tasks require conversion in deeper networks. Our proposed method does not force the main branch to retain complete original information but updates it by generating useful gradient through the auxiliary supervision mechanism. The advantage of this design is that the proposed method can also be applied to shallower networks.

Finally, since auxiliary reversible branch can be removed during the inference phase, the inference capabilities of the original network can be retained. One can also choose any reversible architectures in PGI to play the role of auxiliary reversible branch.

### Multi-level Auxiliary Information

The deep supervision architecture including multiple prediction branch is shown in figure above (c). For object detection, different feature pyramids can be used to perform different tasks, for example together they can detect objects of different sizes. Therefore, after connecting to the deep supervision branch, the shallow features will be guided to learn the features required for small object detection, and at this time the system will regard the positions of objects of other sizes as the background. However, the above deed will cause the deep feature pyramids to lose a lot of information needed to predict the target object. Regarding this issue, authors believe that each feature pyramid needs to receive information about all target objects so that subsequent main branch can retain complete information to learn predictions for various targets.

The concept of multi-level auxiliary information is to insert an integration network between the feature pyramid hierarchy layers of auxiliary supervision and the main branch, and then uses it to combine returned gradients from different prediction heads, as shown in figure above (d). Multi-level auxiliary information is then to aggregate the gradient information containing all target objects, and pass it to the main branch and then update parameters. At this time, the charac- teristics of the main branch’s feature pyramid hierarchy will not be dominated by some specific object’s information. As a result, the proposed method can alleviate the broken information problem in deep supervision. In addition, any integrated network can be used in multi-level auxiliary information. Therefore, one can plan the required semantic levels to guide the learning of network architectures of different sizes.

## Generalized-ELAN (GELAN)

<p align="center">
  <img src="" alt="yolo_v9_gelan" height="350"/>
</p>

By combining two neural network architectures, _CSPNet_ and _ELAN_, which are designed with gradient path planning, authors designed **Generalized Efficient Layer Aggregation Network (GELAN)** that takes into account lighweight, inference speed, and accuracy. Its overall architecture is shown in figure above. Authors generalized the capability of ELAN, which originally only used stacking of convolutional layers, to a new architecture that can use **any** computational blocks.
