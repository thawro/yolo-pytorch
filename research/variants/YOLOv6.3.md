# YOLOv6.3

2023 | [paper](https://arxiv.org/pdf/2301.05586.pdf) | _YOLOv6 v3.0: A Full-Scale Reloading_

In this paper authors renovate the network design and the training strategy. The new features of YOLOv6 are summarized as follows:

* Renew the neck of the detector with a **_Bi-directional Concatenation (BiC)_** module to provide more accurate localization signals. SPPF is simplified to form the SimCSPSPPF Block, which brings performance gains with negligible speed degradation.

* Propose an anchor-aided training (AAT) strategy to enjoy the advantages of both anchor-based and anchor-free paradigms without touching inference efficiency.

* Deepen YOLOv6 to have another stage in the backbone and the neck, which reinforces it to hit a new SoTA performance on the COCO dataset at a high-resolution input.

* Involve a new self-distillation strategy to boost the performance of small models of YOLOv6, in which the heavier branch for DFL is taken as an enhanced auxiliary regression branch during training and is removed at inference to avoid the marked speed decline

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/b43cbce6-08c2-45c1-83a5-c6f1dfc07756" alt="yolo_63" height="300"/>
</p>

## Network Design

In practice, feature integration at multiple scales has been proven to be a critical and effective component of object detection. Feature Pyramid Network (FPN) is proposed to aggregate the high-level semantic features and low-level features via a top-down pathway, which provides more accurate localization. Subsequently, there have been several works on Bi-directional FPN in order to enhance the ability of hierarchical feature representation. PANet adds an extra bottom-up pathway on top of FPN to shorten the information path of low-level and top-level features, which facilitates the propagation of accurate signals from low-level features. BiFPN introduces learnable weights for different input features and simplifies PAN to achieve better performance with high efficiency. PRB-FPN is proposed to retain high-quality features for accurate localization by a parallel FP structure with bi-directional fusion and associated improvements Motivated by the above works, authors design an enhanced-PAN as the detection neck. In order to augment localization signals without bringing in excessive computation burden, authors propose a **_Bi-directional Concatenation (BiC)_** module to integrate feature maps of three adjacent layers, which fuses an extra low-level feature from backbone $C_{i−1}$ into $P_i$ (figure above). In this case, more accurate localization signals can be preserved, which is significant for the localization of small objects.

Moreover, authors simplify the SPPF block (from YOLOv5) to have a CSP-like version called SimCSPSPPF Block, which strengthens the representational ability. Particularly, they revise the SimSPPCSPC Block (from YOLOv7) by shrinking the channels of hidden layers and retouching space pyramid pooling. In addition, they upgrade the CSPBlock with RepBlock (for small models) or CSPStackRep Block (for large models) and accordingly adjust the width and depth. The neck of YOLOv6 is denoted as RepBi-PAN, the framework of which is shown in figure above.

## Anchor-Aided Training

YOLOv6 is an anchor-free detector to pursue a higher inference speed. However, authors experimentally find that the anchor-based paradigm brings additional performance gains on YOLOv6-N under the same settings when compared with the anchor-free one. Moreover, anchor-based ATSS is adopted as the warm-up label as- signment strategy in the early versions of YOLOv6, which stabilizes the training. In light of this, authors propose **_Anchor-Aided Training (AAT)_**, in which the anchor-based auxiliary branches are introduced to combine the advantages of anchor-based and anchor-free paradigms. And they are applied both in the classification and the regression head. Figure below shows the detection head with the auxiliaries.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/9fcfd96f-5505-443c-b38d-8e913536a33c" alt="yolo_63_head" height="400"/>
</p>

During the training stage, the auxiliary branches and the anchor-free branches learn from independent losses while signals are propagated altogether. Therefore, additional embedded guidance information from auxiliary branches is integrated into the main anchor-free heads. Worth mentioning that the auxiliary branches are removed at inference, which boosts the accuracy performance without decreasing speed.

## Self-distillation

In early versions of YOLOv6, the self-distillation is only introduced in large models (i.e., YOLOv6-M/L), which applies the vanilla knowledge distillation technique by minimizing the KL-divergence between the class prediction of the teacher and the student. Meanwhile DFL is adopted as regression loss to perform self-distillation on box regression. The knowledge distillation loss is formulated as:

$$ L_{KD} = KL(p_t^{cls} || p_s^{cls}) + KL(p_t^{reg} || p_s^{reg}) $$

where $p_t^{cls}$ and $p_s^{cls}$ are class prediction of the teacher model and the student model respectively, and accordingly $p_t^{reg}$ and $p_s^{reg}$ are box regression predictions. The overall loss function is now formulated as:

$$ L_{total} = L_{det} + αL_{KD} $$

where $L_{det}$ is the detection loss computed with predictions and labels. The hyperparameter $α$ is introduced to balance two losses. In the early stage of training, the soft labels from the teacher are easier to learn. As the training continues, the performance of the student will match the teacher so that the hard labels will help students more. Upon this, authors apply cosine weight decay to $α$ to dynamically adjust the information from hard labels and soft ones from the teacher. The formulation of $α$ is:

$$ α = −0.99 ∗ \frac{1 − cos(π ∗ \frac{E_i}{E_{max}})}{2} + 1 $$

where $E_i$ denotes the current training epoch and $E_{max}$ represents the maximum training epochs.

Notably, the introduction of DFL requires extra parameters for the regression branch, which affects the inference speed of small models significantly. Therefore, authors specifically design the **_Decoupled Localization Distillation (DLD)_** for their small models to boost performance without speed degradation. Specifically, they append a heavy auxiliary enhanced regression branch to incorporate DFL. During the self-distillation, the student is equipped with a naive regression branch and the enhanced regression branch while the teacher only uses the auxiliary branch. Note that the naive regression branch is only trained with hard labels while the auxiliary is updated according to signals from both the teacher and hard labels. After the distillation, the naive regression branch is retained whilst the auxiliary branch is removed. With this strategy, the advantages of the heavy regression branch for DFL in distillation is considerably maintained without impacting the inference efficiency.
