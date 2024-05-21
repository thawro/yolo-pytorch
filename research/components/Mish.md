# Mish

2019 | [paper](https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf) | _Mish: A Self Regularized Non-Monotonic Neural Activation Function_

The main contribution of this paper is the introduction of **Mish**, a novel neural activation function that outperforms both ReLU and Swish in terms of accuracy across various deep learning benchmarks. Mish is defined as a smooth and non-monotonic activation function, which combines the properties of self-gating and regularization. It is bounded below and unbounded above, making it suitable for avoiding saturation and providing strong regularization effects. Mish's smoothness and non-monotonic nature contribute to effective optimization and generalization in neural networks. Additionally, Mish's consistent improvement in accuracy over Swish and ReLU, as demonstrated in experiments on challenging datasets like CIFAR-10 and CIFAR-100, showcases its robustness and efficiency in enhancing deep learning models

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/bd26222b-6d56-4c2d-99fb-4eaabd271b0a" alt="mish" height="350"/>
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/e21b25d3-58ed-42f6-b632-ee12a0678a6a" alt="mish_derivs" height="350"/>
</p>

The concept of non-linearity in a Neural Network is introduced by an activation function which serves an integral role in the training and performance evaluation of the network. Over the years of theoretical research, many activation functions have been proposed, however, only a few are widely used in mostly all applications which include ReLU (Rectified Linear Unit), TanH (Tan Hyperbolic), Sigmoid, Leaky ReLU and Swish. In this work, a novel, **smooth** and **non-monotonic** neural activation function, **Mish** is proposed, which can be defined as:

$$ f(x) = x * tanh(softplus(x)) $$

Like both Swish and ReLU, Mish is bounded below and unbounded above with a range $[≈-0.31, ∞)$. Mish takes inspiration from Swish by using a property called Self Gating, where the scalar input is provided to the gate. The property of Self-gating is advantageous for replacing activation functions like ReLU (point-wise functions) which take in a single scalar input without requiring to change the network parameters

Although it’s difficult to explain the reason why one activation function performs better than another due to many other training factors, the properties of Mish like being **unbounded above**, **bounded below**, **smooth** and **non-monotonic**, all play a significant role in the improvement of results. Figure below shows the different commonly used activation functions along with the graph of Mish activation for comparison.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/6f54bc11-4e3e-466a-aba3-6bf623cc2634" alt="mish_vs_all" height="350"/>
</p>

* **unbounded above** - a desirable property for any activation function since it avoids saturation which generally causes training to drastically slow down due to near-zero gradients
* **bounded below** - also advantageous since it results in strong regularization effects
* **non-monotonic** - causes small negative inputs to be preserved as negative outputs, which improves expressivity and gradient flow
* **order of continuity being infinite** - also a benefit over ReLU since ReLU has an order of continuity as 0 which means it’s not continuously differentiable causing some undesired problems in gradient-based optimization
* **smooth function** - it helps with effective optimization and generalization. 

The output landscape of 5 layer randomly initialized neural network was compared for ReLU, Swish, and Mish. The observation as shown in figure below, clearly depicts the sharp transition between the scalar magnitudes for the coordinates of ReLU as compared to Swish and Mish. Smoother transition results in smoother loss functions which are easier to optimize and hence the network generalizes better which partially explains why Mish outperforms ReLU. However, in this regards, Mish and Swish are extremely similar in their corresponding output landscapes.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/20fbab38-7892-4418-8501-f7dd82137a95" alt="mish_landscape" height="350"/>
</p>

> **_NOTE:_** Most of the new YOLO approaches use either **Mish** or **Swish** as actiaction functions.
