# Generalized Focal Loss

2020 | [paper](https://arxiv.org/pdf/2006.04388) | _Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection_

OOne-stage detector basically formulates object detection as dense classification and localization (i.e., bounding box regression). The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including:

* the inconsistent usage of the quality estimation and classification between training and inference (i.e., separately trained but compositely used in test)
* the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty which is often the case in complex scenes. 

To address the problems, authors design new representations for these elements. Specifically, they merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain continuous labels, which is beyond the scope of Focal Loss. Authors then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the continuous version for successful optimization

## Focal Loss (FL)

The original FL is proposed to address the one-stage object detection scenario where an extreme imbalance between foreground and background classes often exists during training. A typical form of FL is as follows (we ignore $α_t$ in original paper for simplicity):

$$
FL(p) = -(1 - p_t)^{\gamma} log(p_t), \space \space \space p_t = 
\begin{cases}
p, \space \space \space \textrm{when} \space  y = 1 \\
1 - p, \space \textrm{when} \space y = 0
\end{cases}
$$

where $y ∈ {1, 0}$ specifies the ground-truth class and $p ∈ [0, 1]$ denotes the estimated probability for the class with label $y = 1$. $γ$ is the tunable focusing parameter. Specifically, FL consists of a standard cross entropy part $−log(p_t)$ and a dynamically scaling factor part $(1 − p_t)^γ$ , where the scaling factor $(1 − p_t)^γ$ automatically down-weights the contribution of easy examples during training and rapidly focuses the model on hard examples.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/40797c91-4963-4bfb-924e-51c2d7f22af7" alt="GFL" height="350"/>
</p>

## Quality Focal Loss (QFL)

To solve the inconsistency problem between training and test phases, authors present a joint representation of localization quality (i.e., IoU score) and classification score (“classification-IoU” for short), where its supervision softens the standard one-hot category label and leads to a possible float target $y ∈ [0, 1]$ on the corresponding category. Specifically, $y = 0$ denotes the negative samples with 0 quality score, and $0 < y ≤ 1$ stands for the positive samples with target IoU score $y$. Note that the localization quality label $y$ follows the conventional definition: IoU score between the predicted bounding box and its corresponding ground-truth bounding box during training, with a dynamic value being $0∼1$. Authors adopt the multiple binary classification with sigmoid operators $σ(·)$ for multi-class implementation. For simplicity, the output of sigmoid is marked as $σ$.

Since the proposed classification-IoU joint representation requires dense supervisions over an entire image and the class imbalance problem still occurs, the idea of FL must be inherited. However, the current form of FL only supports $\{1, 0\}$ discrete labels, but the new labels contain decimals. Therefore, authors propose to extend the two parts of FL for enabling the successful training under the case of joint representation: 

* the cross entropy part $-log(p_t)$ is expanded into its complete version $−((1 − y) log(1 − σ) + y log(σ))$
* the scaling factor part $(1 − p_t)^γ$ is generalized into the absolute distance between the estimation $σ$ and its continuous label $y$, i.e., $|y − σ|^β$ ($β ≥ 0$), here $| · |$ guarantees the non-negativity. 

Subsequently, authors combine the above two extended parts to formulate the complete loss objective, which is termed as **Quality Focal Loss (QFL)**:

$$ QFL(σ) = -|y - σ|^β ((1 - y) log(1 - σ) + ylog(σ)) $$

Note that $σ = y$ is the global minimum solution of QFL. Under quality label $y = 0.5$. Similar to FL, the term $∣y − σ∣^β$ of QFL behaves as a modulating factor: when the quality estimation of an example is inaccurate and deviated away from label $y$, the modulating factor is relatively large, thus it pays more attention to learning this hard example. As the quality estimation becomes accurate, i.e., $σ → y$, the factor goes to 0 and the loss for well-estimated examples is down-weighted, in which the parameter $β$ controls the down-weighting rate smoothly ($β = 2$ works best for QFL in the experiments).

## Distribution Focal Loss (DFL)

Authors adopt the relative offsets from the location to the four sides of a bounding box as the regression targets. Conventional operations of bounding box regression model the regressed label $y$ as Dirac delta distribution $δ(x − y)$, where it satisfies $\int_{-\infty}^{+\infty}δ(x − y)dx = 1$ and is usually implemented through fully connected layers. More formally, the integral form to recover $y$ is as follows

$$ y = \int_{-\infty}^{+\infty}δ(x − y)xdx $$

instead of the Dirac delta or Gaussian assumptions, authors propose to directly learn the underlying General distribution $P(x)$ without introducing any other priors. Given the range of label $y$ with minimum $y_0$ and maximum $y_n$ ($y_0 ≤ y ≤ yn, n ∈ N^+$), we can have the estimated value $\hat{y}$ from the model ($\hat{y}$ also meets $y0 ≤ \hat{y} ≤ y_n$):

$$ \hat{y} = \int_{-\infty}^{+\infty} P(x)xdx = \int_{y_0}^{y_n} P(x)xdx $$

To be consistent with convolutional neural networks, we convert the integral over the continuous domain into a discrete representation, via discretizing the range $[y_0, y_n]$ into a set {$y_0$, $y_1$, ..., $y_i$, $y_{i+1}$, ..., $y_{n−1}$, $y_n$} with even intervals $∆$ (authors use $∆ = 1$ for simplicity). Consequently, given the discrete distribution property $\sum_{i=0}^n P(y_i) = 1$, the estimated regression value $\hat{y}$ can be presented as:

$$ \hat{y} = \sum_{i=1}^n P(y_i)y_i $$

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/56960320-b357-4478-98ef-d169fe6cb376" alt="GFL_distributions_comparison" height="200"/>
</p>

As a result, $P(x)$ can be easily implemented through a softmax $S(·)$ layer consisting of $n + 1$ units, with $P(y_i)$ being denoted as $S_i$ for simplicity. Note that $\hat{y}$ can be trained in an end-to-end fashion with traditional loss objectives like SmoothL1, IoU Loss or GIoU Loss. However, there are infinite combinations of values for $P(x)$ that can make the final integral result being $y$, as shown in figure above, which may reduce the learning efficiency. Intuitively compared against (1) and (2), distribution (3) is compact and tends to be more confident and precise on the bounding box estimation, which motivates to optimize the shape of $P(x)$ via explicitly encouraging the high probabilities of values that are close to the target $y$. Furthermore, it is often the case that the most appropriate underlying location, if exists, would not be far away from the coarse label. Therefore, authors introduce the **Distribution Focal Loss (DFL)** which forces the network to rapidly focus on the values near label $y$, by explicitly enlarging the probabilities of $y_i$ and $y_{i+1}$ (nearest two to $y$, $y_i ≤ y ≤ y_{i+1}$). As the learning of bounding boxes are only for positive samples without the risk of class imbalance problem, authors simply apply the complete cross entropy part in QFL for the definition of DFL:

$$ DFL(S_i, S_{i+1}) = -((y_{i+1} - y)log(S_i) + (y - y_i)log(S_{i+1})) $$

Intuitively, DFL aims to focus on enlarging the probabilities of the values around target $y$ (i.e., $y_i$ and $y_{i+1}$). The global minimum solution of DFL, i.e, 

$$ S_i = \frac{y_{i+1} - y}{y_{i+1} - y_i} $$

$$ S_{i+1} = \frac{y - y_i}{y_{i+1} - y_i} $$

, can guarantee the estimated regression target $\hat{y}$ infinitely close to the corresponding label $y$, i.e.

$$ \hat{y} = \sum_{j=0}^n P(y_j)y_j = S_i y_i + S_{i+1} y_{i+1} = \frac{y_{i+1} - y}{y_{i+1} - y_i} y_i + \frac{y - y_i}{y_{i+1} - y_i} y_{i+1} = y $$

, which also ensures its correctness as a loss function.

## Generalized Focal Loss (GFL)

Note that QFL and DFL can be unified into a general form, which is called the **Generalized Focal Loss (GFL)** in the paper. Assume that a model estimates probabilities for two variables $y_l$, $y_r$ ( $y_l < y_r$ ) as $p_{yl}$ , $p_{yr}$ ( $p_{yl} ≥ 0$, $p_{yr} ≥ 0$, $p_{yl} + p_{yr} = 1$ ), with a final prediction of their linear combination being $\hat{y} = y_l p_{yl}  + y_r p_{yr}$ ( $y_l ≤ \hat{y} ≤ y_r$ ). The corresponding continuous label $y$ for the prediction $\hat{y}$ also satisfies $y_l ≤ y ≤ y_r$. Taking the absolute distance $|y − \hat{y}|^β$ ($β ≥ 0$) as modulating factor, the specific formulation of GFL can be written as:

$$ GFL(p_{yl}, p_{yr}) = - |y - (y_l p_{yl}  + y_r p_{yr})|^β ((y_r - y) log(p_{yl}) + (y - y_l)log(p_{yr})) $$

### Properties of GFL

$GFL(p_{yl} , p_{yr})$ reaches its global minimum with 

$$ p^\*_{yl} = \frac{y_r − y}{y_r − y_l} $$

$$ p^\*_{yr} = \frac{y − y_l}{y_r − y_l} $$

which also means that the estimation $\hat{y}$ perfectly matches the continuous label $y$, i.e.

$$ \hat{y} = y_l p^\*_{yl} + y_r p^\*_{yr} = y $$

Obviously, the original FL and the proposed QFL and DFL are all special cases of GFL. Note that GFL can be applied to any one-stage detectors. The modified detectors differ from the original detectors in two aspects:

* during inference, authors directly feed the classification score (joint representation with quality estimation) as NMS scores without the need of multiplying any individual quality prediction if there exists (e.g., centerness as in FCOS and ATSS)
* the last layer of the regression branch for predicting each location of bounding boxes now has $n + 1$ outputs instead of 1 output, which brings negligible extra computing cost

### Training Dense Detectors with GFL

We define training loss L with GFL

$$ L = \frac{1}{N_{pos}} \sum_{z} L_Q + \frac{1}{N_{pos}} \sum_{z} 1_{c^\*_z > 0} (λ_0 L_B + λ_1 L_D) $$

where $L_Q$ is QFL and $L_D$ is DFL. Typically, $L_B$ denotes the GIoU Loss. $N_{pos}$ stands for the number of positive samples. $λ_0$ (typically 2 as default, similarly in) and $λ_1$ (practically $1/4$, averaged over four directions) are the balance weights for $L_Q$ and $L_D$, respectively. The summation is calculated over all locations $z$ on the pyramid feature maps. $1_{c^\*_z > 0}$ is the indicator function, being 1 if $c^\*_z > 0$ and 0 otherwise. Following the common practices in the official codes, authors also utilize the quality scores to weight $L_B$ and $L_D$ during training. 
