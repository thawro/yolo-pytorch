# Effective Squeeze and Excitation

2020 | [paper](https://arxiv.org/pdf/1911.06667v6) | _CenterMask : Real-Time Anchor-Free Instance Segmentation_

Authors propose a simple yet efficient anchor-free instance segmentation, called CenterMask, that adds a novel spatial attention-guided mask (SAG-Mask) branch to anchor-free one stage object detector (FCOS) in the same vein with Mask R-CNN. Plugged into the FCOS object detector, the SAG-Mask branch predicts a segmentation mask on each detected box with the spatial attention map that helps to focus on informative pixels and suppress noise. Authors also present an improved backbone networks, VoVNetV2, with two effective strategies: (1) residual connection for alleviating the optimization problem of larger VoVNet and (2) effective Squeeze-Excitation (eSE) dealing with the channel information loss problem of original SE.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/28a5ad9b-3fe8-4cdb-8ba3-4ac2e11b81e0" alt="CenterMask" height="300"/>
</p>

CenterMask is an instance segmentation architecture composed of VoVNetV2 backbone with FPN as neck and equiped with FCOS detection head. On top of that there is a SAG-Mask component used to predict the segmentation masks on the FCOS-proposed RoI. The procedure of masking objects is composed of detecting objects from the FCOS box head and then predicting segmentation masks inside the cropped regions in a per-pixel manner.

The most important part in terms of object detection applications is the Effective Squeeze-Excitation (eSE), which improves the original SE. 

## Effective Squeeze-Excitation (eSE)

Squeeze-Excitation (SE) is a representative channel attention method adopted in CNN architectures, explicitly models the interdependency between the channels of feature maps to enhance its representation. The SE module squeezes the spatial dependency by global average pooling to learn a channel specific descriptor and then two fully-connected (FC) layers followed by a sigmoid function are used to rescale the input feature map to highlight only useful channels. In short, given input feature map $X_i ∈ R^{C × W × H}$, the channel attention map $A_{ch}(X_i) ∈ R^{C × 1 × 1}$ is computed as:

$$ A_{ch}(X_i) = σ(W_C (δ(W_{C/r} (F_{gap}(X_i)))) $$

where 

$$ F_{gap}(X) = \frac{1}{WH} \sum_{i,j=1}^{W,H} X_{i, j} $$ 

is channel-wise global average pooling, $W_{C/r} , W_C ∈ R^{C × 1 × 1}$ are weights of two fully-connected layers, $δ$ denotes ReLU non-linear operator and $σ$ indicates sigmoid function. However, it is assumed that the SE module has a limitation: channel information loss due to dimension reduction. For avoiding high model complexity burden, two FC layers of the SE module need to reduce channel dimension. Specifically, while the first FC layer reduces input feature channels $C$ to $C/r$ using reduction ratio $r$, the second FC layer expands the reduced channels to original channel size $C$. As a result, this channel dimension reduction causes channel information loss.

Therefore, authors proposed effective SE (eSE) that uses only one FC layer with $C$ channels instead of two FCs without channel dimension reduction, which rather maintains channel information and in turn improves performance. the eSE process is defined as:

$$ A_{eSE}(X_{div} ) = σ(W_C (F_{gap}(X_{div}))) $$

$$ X_{refine} = A_{eSE}(X_{div}) ⊗ X_{div} $$

where $X_{div} ∈ R^{C × W × H}$ is the diversified feature map computed by $1 × 1$ conv in OSA module. As a channel attentive feature descriptor, the $A_{eSE} ∈ R^{C × 1 × 1}$ is applied to the diversified feature map $X_{div}$ to make the diversified feature more informative. Finally, when using the residual connection, the input feature map is element-wise added to the refined feature map $X_{refine}$. The details of How the eSE module is plugged into the OSA module are shown in
figure below.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/51fc37c6-4fbd-4bd7-8cea-a3dacd854d8f" alt="CenterMask_OSA" height="300"/>
</p>
