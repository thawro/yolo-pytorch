
# YOLOv4
2020 | [paper](https://arxiv.org/pdf/2004.10934.pdf) | _YOLOv4: Optimal Speed and Accuracy of Object Detection_

YOLOv4 addresses the need for real-time object detection systems that can achieve high accuracy while maintaining fast inference speeds and possibility to train the object detector on a single customer GPU. Authors of the paper put a great emphasis on analyzing and evaluating the possible training and architectural choices for object detector training and architecture. Two sets of concepts were evaluated separately, that is the Bag of Freebies ([**BoF**](#bag-of-freebies)) and Bag of Specials ([**BoS**](#bag-of-specials)). 

Authors noticed that the reference model which is optimal for classification is not always optimal for a detector. In contrast to the classifier, the detector requires the following:
* Higher input network size (resolution) – for detecting multiple small-sized objects
* More layers – for a higher receptive field to cover the increased size of input network
* More parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image
They also noticed that the influence of the receptive field with different sizes is summarized as follows:
* Up to the object size - allows viewing the entire object
* Up to network size - allows viewing the context around the object
* Exceeding the network size - increases the number of connections between the image point and the final activation

## How it works

Different settings of BoF and BoS were tested for two backbones, that is CSPResNext50 and CSPDarknet53 (CSP module applied to ResNext50 and Darknet53 architectures). The settings include:
* Activations: ReLU, leaky-ReLU, Swish, or Mish
* bbox regression loss: MSE, IoU, GIoU, CIoU, DIoU
* Data augmentation: CutOut, MixUp, CutMix, Mosaic, Self-Adversarial Training
* Regularization method: DropBlock
* Normalization of the network activations by their mean and variance: Batch Normalization ([BN](https://arxiv.org/pdf/1502.03167)), Cross-Iteration Batch Normalization ([CBN](https://arxiv.org/pdf/2002.05712))
* Skip-connections: Residual connections, Weighted residual connections, Multiinput weighted residual connections, or Cross stage partial connections (CSP)
* Eliminate grid sensitivity problem - equation used in YOLOv3 to calculate object coordinates had a flaw related to the need of very high absolute values predictions for points on grid. Authors solved this problem through multiplying the sigmoid by a factor exceeding 1.0, so eliminating the effect of grid on which the object is undetectable
* IoU threshold - using multiple anchors for a single ground truth IoU (truth, anchor) > IoU threshold
* Optimized Anchors - using the optimized anchors for training with the 512x512 network resolution

In order to make the designed detector more suitable for training on single GPU, authors made additional design and improvement as follows:
* Introduced a new method of data augmentation - Mosaic, and Self-Adversarial Training (SAT)
	* Mosaic - represents a new data augmentation method that mixes 4 training images. Thus 4 different contexts are while CutMix mixes only 2 input images. This allows detection of objects outside their normal context. In addition, batch normalization calculates activation statistics from 4 different images on each layer. This significantly reduces the need for a large mini-batch size
	* Self-Adversarial Training (SAT) - new data augmentation technique that operates in 2 forward backward stages. In the 1st stage the neural network alters the original image instead of the network weights. In this way the neural network executes an adversarial attack on itself, altering the original image to create the deception that there is no desired object on the image. In the 2nd stage, the neural network is trained to detect an object on this modified image in the normal way
* Selected optimal hyper-parameters while applying genetic algorithms
* Modified some exsiting methods to make YOLOv4 design suitble for efficient training and detection - modified SAM, modified PAN, and Cross mini-Batch Normalization (CmBN)

## Model architecture

YOLOv4 consists of:
* Backbone - CSPDarknet53
* Neck - SPP and PAN
* Head - YOLOv3 head

YOLOv4 Backbone uses:
* BoF: CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing
* BoS: Mish activation, Cross-stage partial connections (CSP), Multiinput weighted residual connections (MiWRC)

YOLOv4  Detector uses:
* BoF: CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid
sensitivity, Using multiple anchors for a single ground truth, [Cosine annealing scheduler](https://arxiv.org/pdf/1608.03983v5), Optimal hyperparameters, Random training shapes
* BoS: Mish activation, modified SPP-block, modified SAM-block, modified PAN path-aggregation block, DIoU-NMS

Modified SPP, PAN and PAN:
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/cf3549cd-b72f-4369-9c86-50feef5ffc42" alt="modified_SPP" height="300"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/3aaee815-a819-41dd-9b0f-a11c0223af92" alt="modified_SAM" height="300"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/72e5c007-20ff-44e9-a42d-b90d0ee2e019" alt="modified_PAN" height="300"/>
</p>


## Training details

**Classification** (ImageNet): 
* training steps is 8,000,000
* batch size and the mini-batch size are 128 and 32, respectively
* polynomial decay learning rate scheduling strategy is adopted with initial learning rate 0.1
* warm-up steps is 1000
* the momentum and weight decay are respectively set as 0.9 and 0.005
* All BoS experiments use the same hyper-parameter as the default setting, and in the BoF experiments, additional 50% training steps are added
* BoF experiments: MixUp, CutMix, Mosaic, Bluring data augmentation, and label smoothing regularization methods
* BoS experiments: LReLU, Swish, and Mish activation function. All experiments are trained with a 1080 Ti or 2080 Ti GPU.

**Object detection** (MS COCO): 
* training steps is 500,500
* step decay learning rate scheduling strategy is adopted with initial learning rate 0.01 and multiply with a factor 0.1 at the 400,000 steps and the 450,000 steps, respectively
* momentum and weight decay are respectively set as 0.9 and 0.0005
* All architectures use a single GPU to execute multi-scale training in the batch size of 64
* mini-batch size is 8 or 4 depend on the architectures and GPU memory limitation
* Except for using genetic algorithm for hyper-parameter search experiments, all other experiments use default setting
* Genetic algorithm used YOLOv3-SPP to train with GIoU loss and search 300 epochs for min-val 5k sets. Searched settings are adopted:  learning rate 0.00261, momentum 0.949, IoU threshold for assigning ground truth 0.213, and loss normalizer 0.07 for genetic algorithm experiments
* For all experiments, only one GPU is used for training, so techniques such as syncBN that optimizes multiple GPUs are not used
* BoF experiments: grid sensitivity elimination, mosaic data augmentation, IoU threshold, genetic algorithm, class label smoothing, cross mini-batch normalization, self-adversarial training, cosine annealing scheduler, dynamic mini-batch size, DropBlock, Optimized Anchors, different kind of IoU losses
* BoS experiments: Mish, SPP, SAM, RFB, BiFPN, and [Gaussian YOLO](https://arxiv.org/pdf/1904.04620)

