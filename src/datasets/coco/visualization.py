import cv2
import numpy as np

from src.utils.image import get_color


def plot_segmentation(
    image: np.ndarray, segments: np.ndarray, overlay_image: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    for i, seg in enumerate(segments):
        c = get_color(i).tolist()
        overlay_image = cv2.fillPoly(overlay_image, [seg], color=c)
        overlay_image = cv2.drawContours(overlay_image, [seg], -1, color=c, thickness=1)
    return cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0)
