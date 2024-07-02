import cv2
import numpy as np
from torch import Tensor

from src.annots.ops import clip_boxes, xywh2xyxy
from src.annots.visuzalization import plot_boxes
from src.base.results import BaseResult
from src.datasets.coco.utils import COCO_OBJ_LABELS


def postprocess_results(
    image: np.ndarray,
    boxes_xyxy: np.ndarray | Tensor,
    scale_xy: tuple[float, float],
    pad_tlbr: tuple[float, float, float, float],
) -> np.ndarray | Tensor:
    dst_h, dst_w = dst_shape[:2]
    src_h, src_w = src_shape[:2]

    scale = min(src_h / dst_h, src_w / dst_w)  # scale  = old / new
    pad_x = round((src_w - dst_w * scale) / 2 - 0.1)
    pad_y = round((src_h - dst_h * scale) / 2 - 0.1)

    boxes_xyxy[..., 0] -= pad_x
    boxes_xyxy[..., 1] -= pad_y
    boxes_xyxy[..., 2] -= pad_x
    boxes_xyxy[..., 3] -= pad_y
    boxes_xyxy[..., :4] /= scale
    return


class DetectionResult(BaseResult):
    pd_boxes: np.ndarray
    pd_conf: np.ndarray
    pd_classes: np.ndarray
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    image_filepath: str

    image: np.ndarray

    def __init__(
        self,
        image: Tensor,
        gt_boxes_xywh: Tensor,
        gt_classes: Tensor,
        preds: Tensor,
        scale_wh: Tensor,
        pad_tlbr: Tensor,
        image_filepath: str,
    ):
        self.image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        preds_npy = preds.cpu().numpy()
        self.gt_boxes = xywh2xyxy(gt_boxes_xywh.cpu().numpy())
        self.gt_classes = gt_classes.cpu().numpy()
        self.pd_boxes = preds_npy[:, :4]
        self.pd_conf = preds_npy[:, 4]
        self.pd_classes = preds_npy[:, 5]
        self.scale_wh = scale_wh.numpy()  # x, y
        self.pad_tlbr = pad_tlbr.numpy()  # top, left, bottom, right
        self.image_filepath = image_filepath
        self.preds_set = False

    @property
    def num_objects(self) -> int:
        return len(self.pd_conf)

    def set_preds(self):
        if self.preds_set:
            return
        h, w = self.image.shape[:2]
        pad_top, pad_left, pad_bottom, pad_right = self.pad_tlbr
        scale_w, scale_h = self.scale_wh
        x1, x2 = pad_left, w - pad_right
        y1, y2 = pad_top, h - pad_bottom
        # inverse padding
        self.image = self.image[y1:y2, x1:x2]
        self.pd_boxes[..., 0] -= pad_left
        self.pd_boxes[..., 1] -= pad_top
        self.pd_boxes[..., 2] -= pad_left
        self.pd_boxes[..., 3] -= pad_top

        h_, w_ = self.image.shape[:2]

        # inverse scaling
        if scale_w != 1 or scale_h != 1:
            h_new, w_new = int(h_ / scale_h), int(w_ / scale_w)
            # INTER_NEAREST is the fastest
            self.image = cv2.resize(self.image, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            self.pd_boxes[..., 0] /= scale_w
            self.pd_boxes[..., 1] /= scale_h
            self.pd_boxes[..., 2] /= scale_w
            self.pd_boxes[..., 3] /= scale_h
        max_h, max_w = self.image.shape[:2]
        self.pd_boxes = clip_boxes(self.pd_boxes, 0, 0, max_w, max_h)
        self.preds_set = True

    def plot(self) -> dict[str, np.ndarray]:
        label_ids = self.pd_classes.tolist()
        labels = [COCO_OBJ_LABELS[int(label)] for label in label_ids]
        boxes_plot = plot_boxes(
            self.image,
            boxes_xyxy=self.pd_boxes,
            classes=labels,
            classes_ids=label_ids,
            conf=self.pd_conf.tolist(),
        )
        return {"detections": boxes_plot}
