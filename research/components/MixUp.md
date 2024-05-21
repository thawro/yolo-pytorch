# MixUp

2018 | [paper](http://arxiv.org/pdf/1710.09412) | _MixUp: Beyond Empirical Risk Minimization_

Large deep neural networks are powerful, but exhibit undesirable behaviors such as memorization and sensitivity to adversarial examples. The paper introduces _**mixup**_, a simple learning principle to alleviate these issues. In essence, _mixup_ trains a neural network on convex combinations of pairs of examples and their labels. By doing so, _mixup_ regularizes the neural network to favor simple linear behavior in-between training examples

The contribution of _mixup_ paper is to propose a generic vicinal distribution:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/fcfb7d1e-8fc7-4a49-a48c-5652314015c8" alt="mixup_hard_eq" height="60"/>
</p>

where $λ ∼ Beta(α, α)$, for $α ∈ (0, ∞)$. In a nutshell, sampling from the _mixup_ vicinal distribution produces virtual feature-target vectors:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/5992a2fd-34a6-4565-aa5c-9687e1dee702" alt="mixup" height="90"/>
</p>

where $(x_i, y_i)$ and $(x_j , y_j)$ are two feature-target vectors drawn at random from the training data, and $λ ∈ [0, 1]$. The mixup hyper-parameter α controls the strength of interpolation between feature-target pairs, recovering the ERM (Empirical Risk Minimization) principle as $α → 0$.

**What is mixup doing?** 

The mixup vicinal distribution can be understood as a form of data augmentation that encourages the model f to behave linearly in-between training examples. Authors of the paper argue that this linear behaviour reduces the amount of undesirable oscillations when predicting outside the training examples. Also, linearity is a good inductive bias from the perspective of Occam’s razor since it is one of the simplest possible behaviors. _mixup_ leads to decision boundaries that transition linearly from class to class, providing a smoother estimate of uncertainty. Experiments show, that the models trained with mixup are more stable in terms of model predictions and gradient norms in-between training samples.

Mixup example:
<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/6f5f48d7-74cd-4da0-b953-191a12521b0e" alt="mixup_example" height="350"/>
</p>

> **_NOTE:_** Most of the YOLO approaches (new ones) use **MixUp** as one of data augmentation techniques for regularization purposes.
