import torch
import torch.nn as nn
from IPython import embed
from torch import Tensor

from src.annots.ops import boxes_iou


def get_gt_candidate_mask(
    gt_boxes_xyxy: Tensor, anchor_points: Tensor, eps: float = 1e-9
) -> Tensor:
    """
    Select the positive anchor center in gt.

    # gt_candidate_mask has 1 for anchor centers idxs which meet the positions of gt_boxes_xyxy
    # gt_candidate_mask[b, gt_idx, anch_idx] = 1 if anchor_points[anch_idx] is inside gt_boxes_xyxy[gt_idx]

    Args:
        anchor_points (Tensor): shape(hw, 2)
        gt_boxes (Tensor): shape(B, O_max, 4)

    Returns:
        (Tensor): shape(B, O_max, hw)
    """
    n_anchors = anchor_points.shape[0]
    batch_size, n_boxes, _ = gt_boxes_xyxy.shape
    x1y1, x2y2 = gt_boxes_xyxy.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    gt_boxes_ltrb = torch.cat((anchor_points[None] - x1y1, x2y2 - anchor_points[None]), dim=2).view(
        batch_size, n_boxes, n_anchors, -1
    )
    gt_candidate_mask = gt_boxes_ltrb.amin(3).gt_(eps)
    return gt_candidate_mask


def select_highest_ious(mask_pos: Tensor, ious: Tensor, O_max: int):
    """
    If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

    Args:
        mask_pos (Tensor): shape(B, O_max, hw)
        ious (Tensor): shape(B, O_max, hw)

    Returns:
        target_gt_idx (Tensor): shape(B, hw)
        fg_mask (Tensor): shape(B, hw)
        mask_pos (Tensor): shape(B, O_max, hw)
    """
    # (B, O_max, hw) -> (B, hw)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_boxes_xyxy
        # mask_multi_gts[b, :, a_idx] has 1's if a_idx has many assigned ground_truths
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, O_max, -1)  # (B, O_max, hw)
        # max_ious_gts_idx[b, a_idx] is equal to gt_idx with highest iou to a_idx
        max_ious_gts_idx = ious.argmax(1)  # (B, hw)

        # is_max_ious[b, gt_idx, :] has 1's for gt_idx which has the highest iou
        is_max_ious = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_ious.scatter_(1, max_ious_gts_idx.unsqueeze(1), 1)  # (B, O_max, hw)

        # mask_pos takes values of is_max_ious if has multi_gts and original mask_pos otherwise
        mask_pos = torch.where(mask_multi_gts, is_max_ious, mask_pos).float()  # (B, O_max, hw)

        # fg_mask[b, a_idx] is 1 if a_idx got assigned (?)
        fg_mask = mask_pos.sum(-2)  # (B, hw)
    # Find each grid serve which gt(index)
    # target_gt_idx[b, a_idx] is gt_idx if a_idx got assigned to gt_idx (?)
    target_gt_idx = mask_pos.argmax(-2)  # (B, hw)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        n_cls (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        n_cls: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.n_cls = n_cls
        self.bg_idx = n_cls
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_boxes_xyxy: Tensor,
        pd_cls_scores: Tensor,
        gt_boxes_xyxy: Tensor,
        gt_classes: Tensor,
        anchor_points: Tensor,
        mask_gt: Tensor,
    ):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        1. Define mask_pos, align_metric and ious
            mask_pos        (B, O_max, hw): mask_pos[b, gt_idx, a_idx] is 1 if a_idx got assigned to gt_idx
            align_metric    (B, O_max, hw): defines TAL metric between gt_idx and a_idx
            ious            (B, O_max, hw): defines iou between gt_idx and a_idx

        2. Pick highest ious gts if more than one got assigned
            target_gt_idx   (B, hw):        target_gt_idx[b, a_idx] is gt_idx if a_idx got assigned to gt_idx (?)
            fg_mask         (B, hw):        fg_mask[b, a_idx] is 1 if a_idx got assigned (?)
            mask_pos        (B, O_max, hw): updated version of mask_pos

        3. Assign targets for each anchor using target_gt_idx
            agt_boxes_xyxy  (B, hw, 4):     defines target boxes_xyxy assigned for each anchor
            agt_classes     (B, hw):        defines classes ids assigned for each anchor
            agt_cls_scores  (B, hw, n_cls)  defines target cls_scores for each anchor calculated as normalized version of aligned_metric from TAL

        4. Normalize agt_cls_scores

        Args:
            pd_boxes_xyxy (Tensor): shape(B, hw, 4)
            pd_cls_scores (Tensor): shape(B, hw, n_cls)
            gt_boxes (Tensor): shape(B, O_max, 4)
            gt_classes (Tensor): shape(B, O_max, 1)
            anchor_points (Tensor): shape(hw, 2)
            mask_gt (Tensor): shape(B, O_max, 1)

        Returns:
            agt_classes (Tensor): shape(B, hw)
            agt_boxes_xyxy (Tensor): shape(B, hw, 4)
            agt_cls_scores (Tensor): shape(B, hw, n_cls)
            fg_mask (Tensor): shape(B, hw)
            target_gt_idx (Tensor): shape(B, hw)
        """
        self.batch_size = pd_cls_scores.shape[0]
        self.O_max = gt_boxes_xyxy.shape[1]

        if self.O_max == 0:
            device = gt_boxes_xyxy.device
            return (
                torch.full_like(pd_cls_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_boxes_xyxy).to(device),
                torch.zeros_like(pd_cls_scores).to(device),
                torch.zeros_like(pd_cls_scores[..., 0]).to(device),
                torch.zeros_like(pd_cls_scores[..., 0]).to(device),
            )

        # 1
        # mask_pos[b, gt_idx, a_idx] is 1 if a_idx got assigned to gt_idx
        mask_pos, align_metric, ious = self.get_mask_pos(
            pd_boxes_xyxy, pd_cls_scores, gt_boxes_xyxy, gt_classes, anchor_points, mask_gt
        )

        # 2
        target_gt_idx, fg_mask, mask_pos = select_highest_ious(mask_pos, ious, self.O_max)

        # 3
        agt_boxes_xyxy, agt_classes, agt_cls_scores = self.get_targets(
            gt_boxes_xyxy, gt_classes, target_gt_idx
        )
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.n_cls)  # (B, hw) -> (B, hw, n_cls)
        agt_cls_scores = torch.where(fg_scores_mask > 0, agt_cls_scores, 0)

        # 4
        # leave only valid entries
        align_metric *= mask_pos
        # find max metric value for each gt_idx among anchors
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # (B, O_max, 1)
        # find max iou value for each gt_idx among anchors
        pos_ious = (ious * mask_pos).amax(dim=-1, keepdim=True)  # (B, O_max, 1)
        # normalize the align metric
        norm_align_metric = (align_metric * pos_ious / (pos_align_metrics + self.eps)).amax(-2)
        # agt_cls_scores[b, a_idx, cls_idx] is equal to normalized align metric for valid entries
        agt_cls_scores = agt_cls_scores * norm_align_metric.unsqueeze(-1)

        return agt_boxes_xyxy, agt_classes, agt_cls_scores, fg_mask.bool(), target_gt_idx

    def get_mask_pos(
        self,
        pd_boxes_xyxy: Tensor,
        pd_cls_scores: Tensor,
        gt_boxes_xyxy: Tensor,
        gt_classes: Tensor,
        anchor_points: Tensor,
        mask_gt: Tensor,
    ):
        """Return mask_pos - mask representing valid gts and topk prediction candidates"""

        # Select candidates based on gts xyxy coordinates - invalids are 0s
        # (B, O_max, hw)
        gt_candidate_mask = get_gt_candidate_mask(gt_boxes_xyxy, anchor_points)
        gt_candidate_mask = gt_candidate_mask * mask_gt

        # Get anchor_align metric, (B, O_max, hw)
        align_metric, ious = self.get_box_metrics(
            pd_boxes_xyxy, pd_cls_scores, gt_boxes_xyxy, gt_classes, gt_candidate_mask
        )
        # Get topk_mask defined by topk metrics (B, O_max, hw)
        mask_topk = self.get_topk_mask(align_metric, mask_gt=mask_gt)

        # Merge all masks to a final mask, (B, O_max, hw)
        mask_pos = mask_topk * gt_candidate_mask

        return mask_pos, align_metric, ious

    def get_box_metrics(
        self,
        pd_boxes_xyxy: Tensor,
        pd_cls_scores: Tensor,
        gt_boxes_xyxy: Tensor,
        gt_classes: Tensor,
        mask: Tensor,
    ):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        align_metric has values of metric (score and iou) in places where mask is True
        align_metric[b, gt_idx, a_idx] = metric calculated for gt at gt_idx and anchor at a_idx

        """
        n_anchors = pd_boxes_xyxy.shape[1]
        mask = mask.bool()  # B, O_max, hw

        shape = [self.batch_size, self.O_max, n_anchors]
        ious = torch.zeros(shape, dtype=pd_boxes_xyxy.dtype, device=pd_boxes_xyxy.device)
        boxes_cls_scores = torch.zeros(
            shape, dtype=pd_cls_scores.dtype, device=pd_cls_scores.device
        )

        # B, O_max
        idxs_batch = torch.arange(self.batch_size).view(-1, 1).expand(-1, self.O_max)
        idxs_cls = gt_classes.squeeze(-1).long()
        # Get the scores of each grid for each gt cls
        # pd_cls_scores: (B, hw, nc) -> (B, O_max, hw)
        # expands pd_cls_scores to O_max and picks scores at gt_classes indices
        boxes_cls_scores[mask] = pd_cls_scores[idxs_batch, :, idxs_cls][mask]
        # pd_boxes_xyxy: (B, hw, 4)   -> (B, 1, hw, 4)   -> (B, O_max, hw, 4)
        # gt_boxes_xyxy: (B, O_max, 4) -> (B, O_max, 1, 4) -> (B, O_max, hw, 4)
        pd_boxes_xyxy = pd_boxes_xyxy.unsqueeze(1).expand(-1, self.O_max, -1, -1)[mask]
        gt_boxes_xyxy = gt_boxes_xyxy.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask]
        ious[mask] = boxes_iou(gt_boxes_xyxy, pd_boxes_xyxy, CIoU=True).squeeze(-1).clamp_(0)

        # align_metric: (B, O_max, hw)
        align_metric = boxes_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        return align_metric, ious

    def get_topk_mask(self, metrics: Tensor, mask_gt: Tensor):
        """
        Select the top-k candidates based on the given metrics and return corresponding mask of shape (B, O_max, hw).

        Args:
            metrics (Tensor): A tensor of shape (B, O_max, hw), where b is the batch size,
                              O_max is the maximum number of objects, and hw represents the
                              total number of anchor points.
            topk_mask (Tensor): An optional boolean tensor of shape (B, O_max, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (B, O_max, hw) containing the selected top-k candidates.
        """
        # (B, O_max, topk)
        mask_gt_topk = mask_gt.expand(-1, -1, self.topk).bool()
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1)
        batch_size, O_max = topk_idxs.shape[:2]

        # set idxs to 0 if gts are invalid (defined by mask_gt)
        topk_idxs.masked_fill_(~mask_gt_topk, 0)

        # (B, O_max, topk, hw) -> (B, O_max, hw)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones((batch_size, O_max, 1), dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid boxes - the ones that accumulated more "ones", that is the ones that got extra zeros in topk_idxs tensor
        # count_tensor.masked_fill_(count_tensor > 1, 0) # old
        # return count_tensor.to(metrics.dtype)
        topk_mask = ((count_tensor == 1) * 1).to(metrics.dtype)
        return topk_mask

    def get_targets(self, gt_boxes_xyxy: Tensor, gt_classes: Tensor, target_gt_idx: Tensor):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_boxes (Tensor): Ground truth bounding boxes of shape (B, O_max, 4).
            gt_classes (Tensor): Ground truth labels of shape (B, O_max, 1), where b is the batch size and O_max is the maximum number of objects.
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive anchor points, with shape (B, hw), where hw is the total number of anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - agt_classes (Tensor): Shape (B, hw), containing the target labels for positive anchor points.
                - agt_boxes_xyxy (Tensor): Shape (B, hw, 4), containing the target bounding boxes for positive anchor points.
                - agt_cls_scores (Tensor): Shape (B, hw, n_cls), containing the target scores for positive anchor points, where n_cls is the number of object classes.
        """

        # Assigned target labels, (B, 1)
        batch_ind = torch.arange(self.batch_size, dtype=torch.int64, device=gt_classes.device)[
            ..., None
        ]
        target_gt_idx = target_gt_idx + batch_ind * self.O_max  # (B, hw)
        agt_classes = gt_classes.long().flatten()[target_gt_idx]  # (B, hw)
        agt_classes.clamp_(0)
        bs, n_anch = agt_classes.shape[:2]

        # Assigned target boxes, (B, O_max, 4) -> (B, hw, 4)
        agt_boxes_xyxy = gt_boxes_xyxy.view(-1, 4)[target_gt_idx]

        # Assigned target scores (one-hot)
        # agt_cls_scores[b, a_idx, cls_idx] is 1 at agt_classes[b, a_idx]
        agt_cls_scores = torch.zeros(
            (bs, n_anch, self.n_cls), dtype=torch.int64, device=agt_classes.device
        )  # (B, hw, n_cls)
        agt_cls_scores.scatter_(2, agt_classes.unsqueeze(-1), 1)
        return agt_boxes_xyxy, agt_classes, agt_cls_scores


def make_anchors(
    preds: list[Tensor] | tuple[Tensor, ...], strides: list[int], grid_cell_offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype, device = preds[0].dtype, preds[0].device
    for i, stride in enumerate(strides):
        h, w = preds[i].shape[-2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
