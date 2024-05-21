# GIoU

2018 | [paper](https://arxiv.org/pdf/1902.09630) | _Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression_

Intersection over Union ($IoU$) is the most popular evaluation metric used in the object detection benchmarks. However, there is a gap between optimizing the commonly used distance losses for regressing the parameters of a bounding box and maximizing this metric value. The optimal objective for a metric is the metric itself. In the case of axis-aligned $2D$ bounding boxes, it can be shown that $IoU$ can be directly used as a regression loss. However, $IoU$ has a plateau making it infeasible to optimize in the case of non-overlapping bounding boxes. In this paper, we address the weaknesses of $IoU$ by introducing a generalized version as both a new loss and a new metric.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/2fa93d74-942c-4021-abea-a9b506c14c88" alt="giou_comparison" height="400"/>
</p>

However, it can be shown that there is not a strong correlation between minimizing the commonly used losses, e.g. $l_n$-norms, defined on parametric representation of two bounding boxes in $2D$ / $3D$ and improving their $IoU$ values. For example, consider the simple $2D$ scenario in figure above, where the predicted bounding box (black rectangle), and the ground truth box (green rectangle), are represented by their top-left and bottom-right corners, i.e. $(x_1, y_1, x_2, y_2)$. For simplicity, let’s assume that the distance, e.g. $l_2$-norm, between one of the corners of two boxes is fixed. Therefore any predicted bounding box where the second corner lies on a circle with a fixed radius centered on the second corner of the green rectangle (shown by a gray dashed line circle) will have exactly the same $l_2$-norm distance from the ground truth box; however their $IoU$ values can be significantly different. The same argument can be extended to any other representation and loss, e.g. figure above (b). It is intuitive that a good local optimum for these types of objectives may not necessarily be a local optimum for $IoU$. Moreover, in contrast to $IoU$ , $l_n$-norm objectives defined based on the aforementioned parametric representations are not invariant to the scale of the problem. To this end, several pairs of bounding boxes with the same level of overlap, but different scales due to e.g. perspective, will have different objective values. In addition, some representations may suffer from lack of regularization between the different types of parameters used for the representation. For example, in the center and size representation, $(x_c, y_c)$ is defined on the location space while $(w, h)$ belongs to the size space. Complexity increases as more parameters are incorporated, e.g. rotation, or when adding more dimensions to the problem. To alleviate some of the aforementioned problems, SoTA object detectors introduce the concept of an anchor box as a hypothetically good initial guess. They also define a non-linear representation to naively compensate for the scale changes. Even with these handcrafted changes, there is still a gap between optimizing the regression losses and $IoU$ values.

In this paper, authors explore the calculation of $IoU$ between two axis aligned rectangles, or generally two axis aligned n-orthotopes, which has a straightforward analytical solution and in contrast to the prevailing belief, $IoU$ in this case can be backpropagated, i.e. it can be directly used as the objective function to optimize. It is therefore preferable to use $IoU$ as the objective function for $2D$ object detection tasks. Given the choice between optimizing a metric itself vs. a surrogate loss function, the optimal choice is the metric itself. However, $IoU$ as both a metric and a loss has a major issue: **if two objects do not overlap, the $IoU$ value will be zero and will not reflect how far the two shapes are from each other**. In this case of non-overlapping objects, if $IoU$ is used as a loss, its gradient will be zero and cannot be optimized.

## Generalized Intersection over Union

Intersection over Union ($IoU$) for comparing similarity between two arbitrary shapes (volumes) $A, B ⊆ S ∈ R^n$ is attained by:

$$ IoU = \frac{|A ∩ B|}{|A ∪ B|} $$

Two appealing features, which make this similarity measure popular for evaluating many $2D$ / $3D$ computer vision tasks are as follows:

* $IoU$ as a distance, e.g. $L_{IoU} = 1−IoU$ , is a metric (by mathematical definition). It means $L_{IoU}$ fulfills all
properties of a metric such as non-negativity, identity of indiscernibles, symmetry and triangle inequality

* $IoU$ is invariant to the scale of the problem. This means that the similarity between two arbitrary shapes $A$ and $B$ is independent from the scale of their space $S$

However, IoU has a major weakness:

* If $|A ∩ B| = 0$, $IoU(A, B) = 0$. In this case, $IoU$ does not reflect if two shapes are in vicinity of each other or
very far from each other

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/29cdd1c3-84d3-4458-8899-ed0e6046f351" alt="giou_algo_1" height="250"/>
</p>

To address this issue, authors propose a general extension to $IoU$ , namely Generalized Intersection over Union $GIoU$. For two arbitrary convex shapes (volumes) $A, B ⊆ S ∈ R^n$, they first find the smallest convex shapes $C ⊆ S ∈ R^n$ enclosing both $A$ and $B$. For comparing two specific types of geometric shapes, $C$ can be from the same type. For example, two arbitrary ellipsoids, $C$ could be the smallest ellipsoids enclosing them. Then we calculate a ratio between the volume (area) occupied by $C$ excluding $A$ and $B$ and divide by the total volume (area) occupied by $C$. This represents a normalized measure that focuses on the empty volume (area) between $A$ and $B$. Finally $GIoU$ is attained by subtracting this ratio from the $IoU$ value. The calculation of $GIoU$ is summarized in Algorithm above. $GIoU$ as a new metric has the following properties: 

* Similar to $IoU$ , $GIoU$ as a distance, e.g. $L_{GIoU} = 1 − GIoU$ , holding all properties of a metric such as non-negativity, identity of indiscernibles, symmetry and triangle inequality.

* Similar to $IoU$ , $GIoU$ is invariant to the scale of the problem.

* $GIoU$ is always a lower bound for $IoU$ , i.e. $∀A, B ⊆ S, GIoU (A, B) ≤ IoU (A, B)$, and this lower bound becomes tighter when $A$ and $B$ have a stronger shape similarity and proximity, i.e. $\lim_{A \to B} GIoU(A, B) = IoU(A, B)$.

* $∀A, B ⊆ S, 0 ≤ IoU(A, B) ≤ 1$, but $GIoU$ has a symmetric range, i.e. $∀A, B ⊆ S$, $−1 ≤ GIoU(A, B) ≤ 1$.
  * similar to $IoU$ , the value 1 occurs only when two objects overlay perfectly, i.e. if $|A ∪ B| = |A ∩ B|$, then $GIoU = IoU = 1$
  * $GIoU$ value asymptotically converges to -1 when the ratio between occupying regions of two shapes, $|A ∪ B|$, and the volume (area) of the enclosing shape $|C|$ tends to zero, i.e. $\lim_{\frac{|A ∪ B|}{|C|} \to 0} GIoU (A, B) = −1$


In summary, this generalization keeps the major properties of $IoU$ while rectifying its weakness. Therefore, $GIoU$ can be a proper substitute for $IoU$ in all performance measures used in $2D$/$3D$ computer vision tasks

### GIoU as Loss for Bounding Box Regression

So far, we introduced $GIoU$ as a metric for any two arbitrary shapes. However as is the case with $IoU$ , there is no analytical solution for calculating intersection between two arbitrary shapes and/or for finding the smallest enclosing convex object for them. Fortunately, for the $2D$ object detection task where the task is to compare two axis aligned bounding boxes, we can show that $GIoU$ has a straightforward solution. In this case, the intersection and the smallest enclosing objects both have rectangular shapes. It can be shown that the coordinates of their vertices are simply the coordinates of one of the two bounding boxes being compared, which can be attained by comparing each vertices’ coordinates using min and max functions. To check if two bounding boxes overlap, a condition must also be checked. Therefore, we have an exact solution to calculate $IoU$ and $GIoU$. Algorithm below shows how to calculate the $IoU$ and $GIoU$ as loss functions.

<p align="center">
  <img src="https://github.com/thawro/yolo-pytorch/assets/50373360/d85ebff7-b728-4ddd-8fe7-a3561136eeb5" alt="giou_algo_2" height="400"/>
</p>
