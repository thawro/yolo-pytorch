# DropBlock

2018 | [paper](https://arxiv.org/pdf/1810.12890.pdf) | _DropBlock: A regularization method for convolutional networks_

Although dropout is widely used as a regularization technique for fully connected layers, it is often less effective for convolutional layers (in most cases it was used at the FC layers of the conv networks). This lack of success of dropout for convolutional layers is perhaps due to the fact that activation units in convolutional layers are spatially correlated so information can still flow through convolutional networks despite dropout. Thus a structured form of dropout is needed to regularize convolutional networks. The main drawback of dropout is that it drops out features randomly. While this can be effective for fully connected layers, it is less effective for convolutional layers, where features are correlated spatially. When the features are correlated, even with dropout, information about the input can still be sent to the next layer, which causes the networks to overfit. This intuition suggests that a more structured form of dropout is needed to better regularize convolutional networks

**DropBlock** is a structured form of dropout, that is particularly effective to regularize convolutional networks. In DropBlock, features in a block, i.e., a contiguous region of a feature map, are dropped together. As DropBlock discards features in a correlated area, the networks must look elsewhere for evidence to fit the data.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d8718292-fb1e-44f8-9103-4034e1b73ade" alt="drop_block" height="350"/>
</p>

DropBlock is inspired by [Cutout](https://arxiv.org/pdf/1708.04552), a data augmentation method where parts of the input examples are zeroed out. DropBlock generalizes Cutout by applying Cutout at every feature map in a convolutional networks. In our experiments, having a fixed zero-out ratio for DropBlock during training is not as robust as having an increasing schedule for the ratio during training. In other words, it’s better to set the DropBlock ratio to be small initially during training, and linearly increase it over time during training.

DropBlock is a simple method similar to dropout. Its main difference from dropout is that it drops contiguous regions from a feature map of a layer instead of dropping out independent random units. Pseudocode of DropBlock is shown below. 

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/eab7cb36-5aad-40bb-8b61-3d06a3e9744f" alt="drop_block_algo" height="450"/>
</p>

DropBlock has two main parameters which are _block\_size_ and $γ$. block\_size_ is the size of the block to be dropped, and $γ$, controls how many activation units to drop. Authors experimented with a shared DropBlock mask across different feature channels or each feature channel with its DropBlock mask. The above Algorithm corresponds to the latter, which tends to work better in the experiments

Applying DropbBlock in skip connections in addition to the convolution layers increases the accuracy. Also, gradually increasing number of dropped units during training leads to better accuracy and more robust to hyperparameter choices

> **_NOTE:_** Some of the YOLO approaches (newer ones) use the idea of **DropBlock** for regularization purposes.
