# Linear LR Scaling

2018 | [paper](https://arxiv.org/pdf/1706.02677) | _Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour_

In this paper, authors empirically show that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are addressed the trained networks exhibit good generalization. Specifically, they show no loss of accuracy when training with large minibatch sizes up to 8192 images. To achieve this result, they adopt a hyperparameter-free linear scaling rule for adjusting learning rates as a function of minibatch size and develop a new warmup scheme that overcomes optimization challenges early in training. The below figure shows an example of results for various mini-batch sizes on the ImageNet dataset.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/97f1964a-2ad7-4b99-9d27-0be8fc7cb4eb" alt="Linear_Scaling_Error" height="400"/>
</p>

The goal of this report is to demonstrate the feasibility of, and to communicate a practical guide to, large-scale training with distributed synchronous stochastic gradient descent (SGD). As an example, authors scale ResNet-50 training, originally performed with a minibatch size of 256 images (using 8 Tesla P100 GPUs, training time is 29 hours), to larger minibatches (see above figure). In particular, they show that with a large minibatch size of 8192, it is possible to train ResNet-50 in 1 hour using 256 GPUs while maintaining the same level of accuracy as the 256 minibatch baseline. While distributed synchronous SGD is now commonplace, no existing results show that generalization accuracy can be maintained with minibatches as large as 8192 or that such high-accuracy models can be trained in such short time.

To tackle this unusually large minibatch size, authors employ a simple and hyper-parameter-free linear scaling rule to adjust the learning rate. To successfully apply this rule, authors present a new warmup strategy, i.e., a strategy of using lower learning rates at the start of training, to overcome early optimization difficulties. Importantly, not only does the approach match the baseline validation error, but also yields training error curves that closely match the small minibatch baseline.

## Large Minibatch SGD

Authors start by reviewing the formulation of Stochastic Gradient Descent (SGD). Consider supervised learning by minimizing a loss $L(w)$ of the form:

$$ L(w) = \frac{1}{|X|}\sum_{x ∈ X}l(x, w) $$

Here $w$ are the weights of a network, $X$ is a labeled training set, and $l(x, w)$ is the loss computed from samples $x ∈ X$ and their labels $y$. Typically $l$ is the sum of a classification loss (e.g., cross-entropy) and a regularization loss on $w$. Minibatch Stochastic Gradient Descent, usually referred to as simply as SGD in recent literature even though it operates on minibatches, performs the following update:

$$ w_{t+1} = w_t - η\frac{1}{n} \sum_{x ∈ B}\nabla l(x, w_t) $$

Here $B$ is a minibatch sampled from $X$ and $n = |B|$ is the minibatch size, $η$ is the learning rate, and $t$ is the iteration index. Note that in practice we use momentum SGD.

### Learning Rates for Large Minibatches

As is shown in comprehensive experiments, authors found that the following learning rate scaling rule is surprisingly effective for a broad range of minibatch sizes:

> _**Linear Scaling Rule**: When the minibatch size is multiplied by k, multiply the learning rate by k._

All other hyper-parameters (weight decay, etc.) are kept unchanged. The linear scaling rule can help to not only match the accuracy between using small and large minibatches, but equally importantly, to largely match their training curves, which enables rapid debugging and comparison of experiments prior to convergence.

**Interpretation** - Consider a network at iteration $t$ with weights $w_t$, and a sequence of $k$ minibatches $B_j$ for $0 ≤ j < k$ each of size $n$. We compare the effect of executing $k$ SGD iterations with small minibatches $B_j$ and learning rate $η$ versus a single iteration with a large minibatch $∪_j B_j$ of size $kn$ and learning rate $\hat{η}$. According to previous equation, after $k$ iterations of SGD with learning rate $η$ and a minibatch size of $n$ we have:

$$ w_{t+k} = w_t - η \frac{1}{n} \sum_{j < k} \sum_{x ∈ B_j} ∇l(x, w_{t+j}) $$

On the other hand, taking a single step with the large minibatch $∪_j B_j$ of size $kn$ and learning rate $\hat{η}$ yields:

$$ \hat{w_{t+1}} = w_t - \hat{η} \frac{1}{kn} \sum_{j < k} \sum_{x ∈ B_j} ∇l(x, w_{t}) $$

As expected, the updates differ, and it is unlikely that $\hat{w_{t+1}} = w_{t+k}$. However, if we could assume $∇l(x, w_t) ≈ ∇l(x, w_{t+j})$ for $j < k$, then setting $\hat{η} = kη$ would yield $\hat{w_{t+1}} ≈ w_{t+k}$, and the updates from small and large minibatch SGD would be similar. Although this is a strong assumption, authors emphasize that if it were true the two updates are similar only if we set $\hat{η} = kη$.

The above interpretation gives intuition for one case where we may hope the linear scaling rule to apply. In the experiments with $\hat{η} = kη$ (and warmup), small and large minibatch SGD not only result in models with the same final accuracy, but also, the training curves match closely. The empirical results suggest that the above approximation might be valid in large-scale, real-world data.

However, there are at least two cases when the condition $∇l(x, w_t) ≈ ∇l(x, w_{t+j})$ will clearly not hold. First, in initial training when the network is changing rapidly, it does not hold. Authors address this by using a warmup phase. Second, minibatch size cannot be scaled indefinitely: while results are stable for a large range of sizes, beyond a certain point accuracy degrades rapidly. This point is as large as ∼8k in ImageNet experiments

### Warmup

**Gradual warmup** - Authors present an alternative warmup that gradually ramps up the learning rate from a small to a large value. This ramp avoids a sudden increase of the learning rate, allowing healthy convergence at the start of training. In practice, with a large minibatch of size $kn$, they start from a learning rate of $η$ and increment it by a constant amount at each iteration such that it reaches $\hat{η} = kη$ after 5 epochs (results are robust to the exact duration of warmup). After the warmup, the original learning rate schedule is used.
