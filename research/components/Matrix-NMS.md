# Matrix-NMS

2020 | [paper](https://arxiv.org/pdf/2003.10152v3) | _SOLOv2: Dynamic and Fast Instance Segmentation_

The Matrix-NMS is motivated by Soft-NMS. Soft-NMS decays the other detection scores as a monotonic decreasing function $f(iou)$ of their overlaps. By decaying the scores according to IoUs recursively, higher IoU detections will be eliminated with a minimum score threshold. However, such process is sequential like traditional Greedy NMS and could not be implemented in parallel.

Matrix-NMS views this process from another perspective by considering how a predicted mask $m_j$ being suppressed. For $m_j$ , its decay factor is affected by:

* the penalty of each prediction $m_i$ on $m_j$ ($s_i > s_j$), where $s_i$ and $s_j$ are the confidence scores
* the probability of $m_i$ being suppressed.

For the first point, the penalty of each prediction $m_i$ on $m_j$ could be easily computed by $f(iou_{i,j})$. For the second one, the probability of $m_i$ being suppressed is not so elegant to be computed. However, the probability usually has positive correlation with the IoUs. So here authors directly approximate the probability by the most overlapped prediction on $m_i$ as:

$$ f(iou_{., i}) = \min_{\forall s_k > s_i} f(iou_{k, i}) $$

To this end, the final decay factor becomes

$$ decay_j = \min_{\forall s_i > s_j} \frac{f(iou_{i, j}}{f(iou_{., i}} $$

and the updated score is computed by $s_j = s_j · decay_j$. Authors consider two most simple decremented functions, denoted as _linear_ $f(iou_{i,j}) = 1 − iou_{i,j}$, and _Gaussian_ $f(iou_{i, j} = exp(-\frac{iou^{2}_{i, j}}{\sigma})$

## Implementation

All the operations in Matrix-NMS could be implemented in one shot without recurrence. We first compute a $N × N$ pairwise IoU matrix for the top $N$ predictions sorted descending by score. Then we get the most overlapping IoUs by column-wise _max_ on the IoU matrix. Next, the decay factors of all higher scoring predictions are computed, and the decay factor for each prediction is selected as the most effect one by column-wise _min_. Finally, the scores are updated by the decay factors. For usage, we just need thresholding and selecting _top-k_ detections as the final predictions.
