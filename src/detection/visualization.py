import cv2
import numpy as np

from src.detection.utils import _boxes_xyxy
from src.utils.image import get_color, put_txt_in_rect


def plot_boxes(
    image: np.ndarray,
    boxes_xyxy: _boxes_xyxy,
    boxes_labels: list[str],
    boxes_labels_ids: list[int] | np.ndarray,
    boxes_conf: list[float] | None = None,
) -> np.ndarray:
    num_objects = len(boxes_xyxy)
    boxes_image = image.copy()
    for idx in range(num_objects):
        label = boxes_labels[idx]
        label_id = boxes_labels_ids[idx]
        color = tuple(get_color(label_id).tolist())
        xmin, ymin, xmax, ymax = boxes_xyxy[idx]
        pt1 = int(xmin), int(ymin)
        pt2 = int(xmax), int(ymax)
        cv2.rectangle(boxes_image, pt1, pt2, color=color, thickness=3)
        info = f"{label}"
        if boxes_conf is not None:
            info += f" {boxes_conf:.2f}"

        boxes_image = put_txt_in_rect(
            boxes_image,
            info,
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale=1,
            thickness=2,
            point=pt1,
            rect_color=color,
        )
    return boxes_image
