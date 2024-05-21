# CopyPaste

2021 | [paper](https://arxiv.org/pdf/2012.07177) | _Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation_

Authors approach for generating new data using Copy-Paste is very simple. They randomly select two images and apply random scale jittering and random horizontal flipping on each of them. Then select a random subset of objects from one of the images and paste them onto the other image. Lastly, adjust the ground-truth annotations accordingly: remove fully occluded objects and update the masks and bounding boxes of partially occluded objects.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d43efe53-41d7-4534-abef-122b49a11f7b" alt="copy_paste" height="300"/>
</p>

The key idea behind the Copy-Paste augmentation is to paste objects from one image to another image. This can lead to a combinatorial number of new training data, with multiple possibilities for:

* choices of the pair of source image from which instances are copied, and the target image on which they are pasted
* choices of object instances to copy from the source image
* choices of where to paste the copied instances on the target image. 

The large variety of options when utilizing this data augmentation method allows for lots of exploration on how to use the technique most effectively. Prior wors adopts methods for deciding where to paste the additional objects by modeling the surrounding visual context. In contrast, authors of this paper find that a simple strategy of randomly picking objects and pasting them at random locations on the target image provides a significant boost on top of baselines across multiple settings. Specifically, it gives solid improvements across a wide range of settings with variability in backbone architecture, extent of scale jittering, training schedule and image size. In combination with large scale jittering, authors show that the Copy-Paste augmentation results in significant gains in the data-efficiency on COCO. 

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/54bd8f27-6aa0-4003-aeef-2b930bd2586c" alt="copy_paste_long_training" height="300"/>
</p>
