import numpy as np
from torch import Tensor

from src.annots.ops import postprocess_boxes
from src.annots.visuzalization import plot_boxes
from src.base.results import BaseResult
from src.datasets.coco.utils import COCO_OBJ_LABELS


class DetectionResult(BaseResult):
    preds: np.ndarray
    image: np.ndarray

    def __init__(self, model_input_image: Tensor, preds: Tensor):
        self.model_input_image = model_input_image.permute(1, 2, 0).numpy()
        self.preds = preds.cpu().numpy()

    def set_preds(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.image = self.model_input_image * std + mean
        self.image = (self.image * 255).astype(np.uint8)
        shape = list(self.image.shape[:2])
        self.preds[:, :4] = postprocess_boxes(self.preds[:, :4], shape, shape)

    def plot(self) -> dict[str, np.ndarray]:
        boxes = self.preds[:, :4]
        boxes_conf = self.preds[:, 4]
        boxes_cls = self.preds[:, 5]
        label_ids = boxes_cls.tolist()
        labels = [COCO_OBJ_LABELS[int(label)] for label in label_ids]
        boxes_plot = plot_boxes(
            self.image,
            boxes_xyxy=boxes,
            classes=labels,
            classes_ids=label_ids,
            conf=boxes_conf.tolist(),
        )
        return {"detections": boxes_plot}
