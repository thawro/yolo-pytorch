import cv2
import numpy as np

_bbox = tuple[int, int, int, int]


def plot_bboxes(image: np.ndarray, bboxes: list[_bbox]) -> np.ndarray:
    for x, y, w, h in bboxes:
        pt1 = x - w // 2, y - h // 2
        pt2 = x + w // 2, y + h // 2
        cv2.rectangle(image, pt1, pt2, color=(30, 240, 30), thickness=3)
    return image
