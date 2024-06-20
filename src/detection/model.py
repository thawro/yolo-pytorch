import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.annots.ops import ltrb2xywh, xywh2xyxy, xyxy2xywh
from src.base.model import BaseImageModel, BaseInferenceModel
from src.detection.architectures.yolo_v10.model import HeadPreds, YOLOv8, YOLOv10
from src.detection.tal import make_anchors
from src.logger.pylogger import log


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for obbs, powered by probiou and fast-nms.

    Args:
        boxes (torch.Tensor): (N, 5), xywhr.
        scores (torch.Tensor): (N, ).
        threshold (float): IoU threshold.

    Returns:
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    box_preds,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    max_nms=30000,
    max_wh=7680,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        box_preds (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        max_det (int): The maximum number of boxes to keep after NMS.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1
    batch_size, num_outs = box_preds.shape[:2]
    nc = num_outs - 4  # number of classes
    nm = num_outs - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = box_preds[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    box_preds = box_preds.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        box_preds[..., :4] = xywh2xyxy(box_preds[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=box_preds.device)] * batch_size
    for xi, x in enumerate(box_preds):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            # sort by confidence and remove excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
    return output


def v10postprocess(boxes_xywh: Tensor, scores: Tensor, max_det: int = 300, num_classes: int = 80):
    print("v10postprocess: ", boxes_xywh.shape, scores.shape)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes_xywh = torch.gather(boxes_xywh, dim=1, index=index.repeat(1, 1, boxes_xywh.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % num_classes
    index = index // num_classes
    boxes_xywh = boxes_xywh.gather(
        dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes_xywh.shape[-1])
    )
    return boxes_xywh, scores, labels


class BaseDetectionModel(BaseImageModel):
    net: YOLOv10 | YOLOv8

    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, net: YOLOv10 | YOLOv8, stride: int):
        super().__init__(net, stride, ["images"], ["detections"])
        self.dynamic = True
        self.stride = torch.Tensor([8, 16, 32])
        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.export = False
        self.format = "torch"
        self.shape = None
        net = net.module if isinstance(net, DDP) else net
        print(type(net))
        if hasattr(net, "head_one2one"):  # YOLOv10
            head = net.head_one2one
        else:  # YOLov8
            head = net.head

        self.num_reg_preds = head.num_reg_preds
        self.num_classes = head.num_classes
        self.num_dfl_preds = self.num_reg_preds * 4
        self.num_outputs = self.num_dfl_preds + self.num_classes
        self.dfl = head.dfl

    def init_weights(self):
        # TODO
        pass

    def example_input(self) -> dict[str, Tensor]:
        return super().example_input(1, 3, 640, 640)


class v8DetectionModel(BaseDetectionModel):
    net: YOLOv8

    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, net: YOLOv8, stride: int):
        super().__init__(net, stride)

    def train_forward(self, images: Tensor) -> HeadPreds:
        return self.net(images)

    def inference_forward(self, images: Tensor) -> tuple[HeadPreds, list[Tensor]]:
        preds = self.net(images)
        shape = preds[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.num_outputs, -1) for xi in preds], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(preds, self.stride, 0.5)
            )
            self.shape = shape

        boxes, cls = x_cat.split((self.num_reg_preds * 4, self.num_classes), 1)
        boxes_xywh = ltrb2xywh(self.dfl(boxes), self.anchors.unsqueeze(0), dim=1) * self.strides
        boxes_preds = non_max_suppression(
            torch.cat((boxes_xywh, cls.sigmoid()), 1),
            conf_thres=0.25,
            iou_thres=0.5,
            agnostic=False,
            max_det=300,
            classes=None,
        )
        return preds, boxes_preds


class v10DetectionModel(BaseDetectionModel):
    net: YOLOv10

    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, net: YOLOv10, stride: int):
        super().__init__(net, stride)

    def train_forward(self, images: Tensor) -> tuple[HeadPreds, HeadPreds]:
        one2one_preds, one2many_preds = self.net(images)
        return one2one_preds, one2many_preds

    def inference_forward(self, images: Tensor) -> tuple[tuple[HeadPreds, HeadPreds], list[Tensor]]:
        one2one_preds, one2many_preds = self.net(images)
        preds = one2one_preds

        shape = preds[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.num_outputs, -1) for xi in preds], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(preds, self.stride, 0.5)
            )
            self.shape = shape

        boxes, cls = x_cat.split((self.num_reg_preds * 4, self.num_classes), 1)
        boxes_xywh = ltrb2xywh(self.dfl(boxes), self.anchors.unsqueeze(0), dim=1) * self.strides
        boxes_xywh, scores, labels = v10postprocess(
            boxes_xywh.swapaxes(-1, -2),
            cls.sigmoid().swapaxes(-1, -2),
            max_det=300,
            num_classes=self.num_classes,
        )
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_preds = torch.cat([boxes_xyxy, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        # TODO
        conf_thr = 0.15
        mask = boxes_preds[..., 4] > conf_thr

        boxes_preds = [p[mask[idx]] for idx, p in enumerate(boxes_preds)]
        return (one2one_preds, one2many_preds), boxes_preds
