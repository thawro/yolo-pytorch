# Cosine Annealing

2017 | [paper](https://arxiv.org/pdf/1608.03983.pdf) | _SGDR: Stochastic Gradient Descent with Warm Restarts_

Cosine Annealing with warm restarts is a type of learning rate schedule that has the effect of starting with a large learning rate that is relatively rapidly decreased to a minimum value before being increased rapidly again. The resetting of the learning rate acts like a simulated restart of the learning process and the re-use of good weights as the starting point of the restart is referred to as a "warm restart" in contrast to a "cold restart" where a new set of small random numbers may be used as a starting point.

Authors of the work considered one of the simplest warm restart approaches. They simulated a new warm-started run / restart of SGD once $T_i$ epochs are performed, where $i$ is the index of the run. Importantly, the restarts are not performed from scratch but emulated by increasing the learning rate $η_t$ while the old value of $x_t$ is used as an initial solution. The amount of this increase controls to which extent the previously acquired information (e.g., momentum) is used. Within the $i$-th run, the learning rate is decayed with a cosine annealing for each batch as follows:

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/a9c1d5c7-2912-451c-bd63-5b0006052bd8" alt="cosine_annealing_eq" height="70"/>
</p>

where $η^i_{min}$ and $η^i_{max}$ are ranges for the learning rate, and $T_{cur}$ accounts for how many epochs have been performed since the last restart. Since $T_{cur}$ is updated at each batch iteration $t$, it can take discredited values such as $0.1$, $0.2$, etc. Thus, $η_t = η^i_{max}$ when $t = 0$ and $T_{cur} = 0$. Once $T_{cur} = T_i$, the cos function will output $−1$ and thus $η_t = η^i_{min}$. The decrease of the learning rate is shown below (left - without restart, right - with restart every ~2k iterations).

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2d2c4971-fd4a-4ab6-a6f8-bc068b60c624" alt="cosine_annealing" height="350"/>
</p>

> **_NOTE:_** Some of the YOLO approaches use the idea of **Cosine Annealing** (without warm restarts) for learning rate scheduling.
