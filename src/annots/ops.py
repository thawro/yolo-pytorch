import math

import numpy as np
import torch
from torch import Tensor


def _empty_like(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, np.ndarray):
        return np.empty_like(x)
    else:
        return torch.empty_like(x)


def xyxy2xywh(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    boxes_ = _empty_like(boxes)
    xmin, ymin = boxes[..., 0], boxes[..., 1]
    xmax, ymax = boxes[..., 2], boxes[..., 3]
    boxes_[..., 0] = (xmin + xmax) / 2  # xc
    boxes_[..., 1] = (ymin + ymax) / 2  # yc
    boxes_[..., 2] = xmax - xmin  # w
    boxes_[..., 3] = ymax - ymin  # h
    return boxes_


def xyxy2xywh_coco(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    boxes_ = _empty_like(boxes)
    xmin, ymin = boxes[..., 0], boxes[..., 1]
    xmax, ymax = boxes[..., 2], boxes[..., 3]
    boxes_[..., 0] = xmin
    boxes_[..., 1] = ymin
    boxes_[..., 2] = xmax - xmin  # w
    boxes_[..., 3] = ymax - ymin  # h
    return boxes_


def xywh2xyxy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    boxes_ = _empty_like(boxes)
    xy = boxes[..., :2]  # centers
    wh = boxes[..., 2:] / 2  # half wh
    boxes_[..., :2] = xy - wh  # xmin, ymin
    boxes_[..., 2:] = xy + wh  # xmax, ymax
    return boxes_


def xyxy2ltrb(boxes_xyxy: Tensor, anchor_points: Tensor) -> Tensor:
    """Transform xyxy boxes to distances (ltrb)."""
    x1y1, x2y2 = boxes_xyxy.chunk(2, -1)
    boxes_ltrb = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    return boxes_ltrb


def segment2box(segment: np.ndarray, height: int, width: int) -> np.ndarray:
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )


def clip_boxes(
    boxes_xyxy: np.ndarray | Tensor, min_x: int, min_y: int, max_x: int, max_y: int
) -> Tensor | np.ndarray:
    if isinstance(boxes_xyxy, Tensor):
        boxes_xyxy[..., 0] = boxes_xyxy[..., 0].clamp(min_x, max_x)
        boxes_xyxy[..., 1] = boxes_xyxy[..., 1].clamp(min_y, max_y)
        boxes_xyxy[..., 2] = boxes_xyxy[..., 2].clamp(min_x, max_x)
        boxes_xyxy[..., 3] = boxes_xyxy[..., 3].clamp(min_y, max_y)
    else:  # numpy (faster grouped)
        boxes_xyxy[..., [0, 2]] = boxes_xyxy[..., [0, 2]].clip(min_x, max_x)
        boxes_xyxy[..., [1, 3]] = boxes_xyxy[..., [1, 3]].clip(min_y, max_y)
    return boxes_xyxy


# def postprocess_boxes(
#     boxes_xyxy: np.ndarray | Tensor, dst_shape: list, src_shape: list
# ) -> np.ndarray | Tensor:
#     dst_h, dst_w = dst_shape[:2]
#     src_h, src_w = src_shape[:2]

#     scale = min(src_h / dst_h, src_w / dst_w)  # scale  = old / new
#     pad_x = round((src_w - dst_w * scale) / 2 - 0.1)
#     pad_y = round((src_h - dst_h * scale) / 2 - 0.1)

#     boxes_xyxy[..., 0] -= pad_x
#     boxes_xyxy[..., 1] -= pad_y
#     boxes_xyxy[..., 2] -= pad_x
#     boxes_xyxy[..., 3] -= pad_y
#     boxes_xyxy[..., :4] /= scale
#     return clip_boxes(boxes_xyxy, 0, 0, dst_w, dst_h)


def boxes_iou(
    box: Tensor,
    boxes: Tensor,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> Tensor:
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    DIoU/CIoU:  https://arxiv.org/abs/1911.08287v1
    GIoU:       https://arxiv.org/pdf/1902.09630.pdf

    CIoU: https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter_area = inter_w * inter_h

    # Union Area
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    # IoU
    iou = inter_area / union_area
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        convex_area = cw * ch + eps  # convex area
        return iou - (convex_area - union_area) / convex_area  # GIoU
    return iou  # IoU


def ltrb2xyxy(boxes_ltrb: Tensor, anchor_points: Tensor, dim: int = -1) -> Tensor:
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = boxes_ltrb.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


def ltrb2xywh(boxes_ltrb: Tensor, anchor_points: Tensor, dim: int = -1) -> Tensor:
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = boxes_ltrb.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((c_xy, wh), dim)
