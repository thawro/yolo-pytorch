import torch
import torch.nn.functional as F
from IPython import embed
from torch import Tensor, nn

from src.annots.ops import boxes_iou, ltrb2xyxy, xywh2xyxy, xyxy2ltrb
from src.detection.architectures.yolo_v10.model import HeadPreds

from .tal import TaskAlignedAssigner, make_anchors


class BoxIoULoss(nn.Module):
    def calculate_loss(
        self,
        pd_boxes_xyxy: Tensor,
        gt_boxes_xyxy: Tensor,
        weight: Tensor,
    ) -> Tensor:
        """IoU loss."""
        iou = boxes_iou(pd_boxes_xyxy, gt_boxes_xyxy, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum()
        return loss_iou


class BoxDFLLoss(nn.Module):
    def __init__(self, n_reg: int):
        super().__init__()
        self.n_reg = n_reg

    def calculate_loss(
        self, pd_boxes_distribution: Tensor, gt_boxes_ltrb: Tensor, weight: Tensor
    ) -> Tensor:
        """DFL loss."""
        loss_dfl = self._df_loss(pd_boxes_distribution.view(-1, self.n_reg), gt_boxes_ltrb)
        loss_dfl = (loss_dfl * weight).sum()
        return loss_dfl

    @staticmethod
    def _df_loss(boxes_distribution: Tensor, gt_boxes_ltrb: Tensor) -> Tensor:
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = gt_boxes_ltrb.long()  # target left
        tr = tl + 1  # target right
        wl = tr - gt_boxes_ltrb  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(boxes_distribution, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(boxes_distribution, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BoxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, n_reg: int):
        super().__init__()
        self.iou_loss_fn = BoxIoULoss()
        self.use_dfl = n_reg > 1
        self.n_reg = n_reg
        if self.use_dfl:
            self.dfl_loss_fn = BoxDFLLoss(n_reg)
        else:
            self.dfl_loss_fn = None

    def calculate_loss(
        self,
        pd_boxes_distribution: Tensor,
        pd_boxes_xyxy: Tensor,
        gt_boxes_xyxy: Tensor,
        gt_cls_scores: Tensor,
        anchor_points: Tensor,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """IoU and DFL loss."""
        # each anchor prediction box loss (iou_loss and dfl) is weighted by its gt cls_score (calculated by TAL)
        weight = gt_cls_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = self.iou_loss_fn.calculate_loss(
            pd_boxes_xyxy[fg_mask], gt_boxes_xyxy[fg_mask], weight
        )
        if self.dfl_loss_fn is not None:
            gt_boxes_ltrb = xyxy2ltrb(gt_boxes_xyxy, anchor_points)
            gt_boxes_ltrb = gt_boxes_ltrb.clamp_(0, self.n_reg - 1 - 0.01)
            loss_dfl = self.dfl_loss_fn.calculate_loss(
                pd_boxes_distribution[fg_mask], gt_boxes_ltrb[fg_mask], weight
            )
        else:
            loss_dfl = torch.tensor(0.0).to(pd_boxes_distribution.device)
        return loss_iou, loss_dfl


class v8DetectionLoss(nn.Module):
    """yolov8 Criterion class for computing training losses."""

    def __init__(
        self,
        n_cls: int = 80,
        n_reg: int = 16,
        tal_topk: int = 10,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        strides: list[int] = [8, 16, 32],
    ):
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        super().__init__()
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.strides = strides
        self.n_cls = n_cls
        self.n_reg = n_reg
        self.n_dfl = n_reg * 4
        self.n_out = self.n_cls + self.n_dfl
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, n_cls=self.n_cls, alpha=0.5, beta=6.0)
        self.box_loss = BoxLoss(n_reg)
        self.proj = torch.nn.Parameter(torch.arange(n_reg, dtype=torch.float), requires_grad=False)

    def preprocess(self, gt_batch_idxs: Tensor, targets: Tensor, batch_size: int) -> Tensor:
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        _, counts = gt_batch_idxs.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
        for j in range(batch_size):
            matches = gt_batch_idxs == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches]
        return out

    def decode_boxes(self, boxes_distribution: Tensor) -> Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        batch_size, num_anchors, num_channels = boxes_distribution.shape
        pd_boxes_xyxy = (
            boxes_distribution.view(batch_size, num_anchors, 4, num_channels // 4)
            .softmax(3)
            .matmul(self.proj.to(boxes_distribution.dtype))
        )
        return pd_boxes_xyxy

    def calculate_loss(
        self, preds: HeadPreds, gt_boxes_xywh: Tensor, gt_classes: Tensor, gt_batch_idxs: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.

        B - batch_size
        O_sum - total number of objects in whole batch (sum of objects O_i in each image)
        O_max - maximum number of objects in an image across batch, calculated as max(O_i)
        n_dfl = 64, n_cls = 80, n_out = n_dfl + n_cls
        hw = 80*80 + 40*40 + 20*20

        1. Flatten preds:
            [(B, n_out, 80, 80), (B, n_out, 40, 40), (B, n_out, 20, 20)] -> (B, n_out, hw)
            raw_preds                                                    -> preds

        2. Split channels into DFL boxes coords distribution and cls scores
            (B, n_out, hw)    -> (B, n_dfl, hw)         and     (B, n_cls, hw)
            preds               -> pd_boxes_distribution    and     pd_cls_scores

        3. Permute boxes distribution and cls scores
            pd_boxes_distribution:  (B, n_dfl, hw) ->  (B, hw, n_dfl)
            pd_cls_scores:          (B, n_cls, hw) ->  (B, hw, n_cls)

        4. Create anchor points and stride tensor for each predicted box
            anchor_points:  (hw, 2)
            strides:        (hw, 1)

        5. Preprocess targets
            gt_boxes_xywh: (O_sum, 4)   ->   gt_boxes_xyxy: (B, O_max, 4)
            gt_classes: (O_sum, 1)      ->   gt_clases: (B, O_max, 1)

        6. Create mask_gt indicating which boxes are going to be considered during loss
        computation - only the "real" ones, that is the ones that were not filled with zeros in step 5
            mask_gt: (B, O_max, 1)

        7. Decode preds - transform boxes_distribution to boxes_ltrb (left/top/right/bottom distances from cell border) and further to boxes_xyxy
            boxes_distribution: (B, hw, n_dfl) -> boxes_ltrb: (B, hw, 4)
            boxes_ltrb -> boxes_xyxy

        8. Assign ground truths to predictions using TAL
            agt_boxes_xyxy    (B, hw, 4)      : defines target boxes_xyxy assigned for each anchor
            agt_classes       (B, hw)         : defines classes ids assigned for each anchor
            agt_cls_scores    (B, hw, n_cls)  : defines target cls_scores for each anchor calculated as normalized version of aligned_metric from TAL
            agt_fg_mask       (B, hw)         : defines which anchors got assigned to gts and thus are taken into account for box loss

        9. Calculate cls_loss with BCE between agt_cls_scores and pd_cls_scores
        10. Calculate box_loss with IoULoss and DFL
            IoULoss between: pd_boxes_xyxy and agt_boxes_xyxy
            DFL between:     pd_boxes_distribution and gt_boxes_ltrb (clamped to n_reg)
        """
        device = preds[0].device
        dtype = preds[0].dtype
        batch_size, _, preds_h, preds_w = preds[0].shape
        img_h, img_w = preds_h * self.strides[0], preds_w * self.strides[0]
        scale_xyxy = torch.tensor([img_w, img_h, img_w, img_h], device=device, dtype=dtype)

        # 1
        preds_flat = torch.cat([xi.view(batch_size, self.n_out, -1) for xi in preds], 2)

        # 2
        pd_boxes_distribution, pd_cls_scores = preds_flat.split((self.n_dfl, self.n_cls), 1)

        # 3
        pd_cls_scores = pd_cls_scores.permute(0, 2, 1).contiguous()
        pd_boxes_distribution = pd_boxes_distribution.permute(0, 2, 1).contiguous()

        # 4
        anchor_points, stride_tensor = make_anchors(preds, self.strides, 0.5)

        # 5
        targets = torch.cat((gt_classes.view(-1, 1), gt_boxes_xywh), 1)
        targets = self.preprocess(gt_batch_idxs, targets, batch_size)
        targets[..., 1:5] = xywh2xyxy(targets[..., 1:5].mul_(scale_xyxy))
        gt_classes, gt_boxes_xyxy = targets.split((1, 4), 2)

        # 6
        mask_gt = gt_boxes_xyxy.sum(2, keepdim=True).gt_(0)  # TODO: amin instead of sum
        # mask_gt defines only which gt_boxes are valid

        # 7
        if self.box_loss.use_dfl:
            pd_boxes_ltrb = self.decode_boxes(pd_boxes_distribution)
        else:
            pd_boxes_ltrb = pd_boxes_distribution
        pd_boxes_xyxy = ltrb2xyxy(pd_boxes_ltrb, anchor_points)

        # 8
        agt_boxes_xyxy, agt_classes, agt_cls_scores, agt_fg_mask, _ = self.assigner(
            (pd_boxes_xyxy.detach() * stride_tensor).to(gt_boxes_xyxy.dtype),
            pd_cls_scores.detach().sigmoid(),
            gt_boxes_xyxy,
            gt_classes,
            anchor_points * stride_tensor,
            mask_gt,
        )
        # agt_boxes_xyxy    (B, hw, 4)      : defines target boxes_xyxy assigned for each anchor
        # agt_classes       (B, hw)         : defines classes ids assigned for each anchor
        # agt_cls_scores    (B, hw, n_cls)  : defines target cls_scores for each anchor calculated as normalized version of aligned_metric from TAL
        # agt_fg_mask       (B, hw)         : defines which anchors got assigned to gts and thus are taken into account for box loss

        agt_cls_scores_sum = max(agt_cls_scores.sum(), 1)  # normalization factor for losses

        # 9
        cls_loss = self.bce(pd_cls_scores, agt_cls_scores.to(dtype)).sum() / agt_cls_scores_sum
        # loss[1] = self.varifocal_loss(pd_cls_scores, agt_cls_scores, agt_classes) / agt_cls_scores_sum

        # 10
        box_loss, dfl_loss = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        if agt_fg_mask.sum():
            agt_boxes_xyxy /= stride_tensor
            box_loss, dfl_loss = self.box_loss.calculate_loss(
                pd_boxes_distribution=pd_boxes_distribution,
                pd_boxes_xyxy=pd_boxes_xyxy,
                gt_boxes_xyxy=agt_boxes_xyxy,
                gt_cls_scores=agt_cls_scores,
                anchor_points=anchor_points,
                fg_mask=agt_fg_mask,
            )
            box_loss /= agt_cls_scores_sum
            dfl_loss /= agt_cls_scores_sum

        box_loss *= self.box_gain
        cls_loss *= self.cls_gain
        dfl_loss *= self.dfl_gain
        final_loss = (box_loss + cls_loss + dfl_loss) * batch_size
        losses = {
            "box_loss": box_loss.item(),
            "cls_loss": cls_loss.item(),
            "dfl_loss": dfl_loss.item(),
        }
        return final_loss, losses


class v10DetectLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        n_reg: int = 16,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        strides: list[int] = [8, 16, 32],
    ):
        super().__init__()
        params = dict(
            num_classes=num_classes,
            n_reg=n_reg,
            box_gain=box_gain,
            cls_gain=cls_gain,
            dfl_gain=dfl_gain,
            strides=strides,
        )
        self.one2many_loss = v8DetectionLoss(**params, tal_topk=10)
        self.one2one_loss = v8DetectionLoss(**params, tal_topk=1)

    def calculate_loss(
        self,
        one2one_preds: HeadPreds,
        one2many_preds: HeadPreds,
        gt_boxes_xywh: Tensor,
        gt_classes: Tensor,
        gt_batch_idxs: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        final_o2m_loss, o2m_losses = self.one2many_loss.calculate_loss(
            one2many_preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
        )
        final_o2o_loss, o2o_losses = self.one2one_loss.calculate_loss(
            one2one_preds, gt_boxes_xywh, gt_classes, gt_batch_idxs
        )
        final_loss = final_o2m_loss + final_o2o_loss
        losses = {}
        for k, v in o2m_losses.items():
            losses[f"o2m_{k}"] = v
        for k, v in o2o_losses.items():
            losses[f"o2o_{k}"] = v

        return final_loss, losses
