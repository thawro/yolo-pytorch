import cv2
import numpy as np

_bbox = tuple[int, int, int, int]


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> _bbox:
    return x, y, x + w, y + h


def plot_boxes(
    image: np.ndarray,
    boxes_xywh: list[_bbox],
    boxes_labels: list[int],
    boxes_conf: list[float] | None,
) -> np.ndarray:
    num_objects = len(boxes_xywh)
    boxes_image = image.copy()
    for idx in range(num_objects):
        x, y, w, h = boxes_xywh[idx]
        label = boxes_labels[idx]
        xmin, ymin, xmax, ymax = xywh_to_xyxy(x, y, w, h)
        pt1 = int(xmin), int(ymin)
        pt2 = int(xmax), int(ymax)
        cv2.rectangle(boxes_image, pt1, pt2, color=(30, 240, 30), thickness=3)
        info = f"{label}"
        if boxes_conf is not None:
            info += f" {boxes_conf:.2f}"
        cv2.putText(
            boxes_image,
            info,
            pt1,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(30, 240, 30),
            thickness=1,
        )
    return boxes_image
