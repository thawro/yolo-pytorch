from typing import Literal

import cv2
import numpy as np

from src.utils.image import get_color, put_txt_in_rect


def plot_segments(image: np.ndarray, segments: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay_image = image.copy()
    for i, seg in enumerate(segments):
        c = get_color(i).tolist()
        overlay_image = cv2.fillPoly(overlay_image, [seg], color=c)
        overlay_image = cv2.drawContours(overlay_image, [seg], -1, color=c, thickness=1)
    return cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0)


def plot_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: list[str] | np.ndarray | None,
    classes_ids: list[int] | np.ndarray | None,
    conf: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    num_objects = len(boxes_xyxy)
    boxes_image = image.copy()
    for idx in range(num_objects):
        label_id, label = "", ""
        if classes_ids is not None:
            label_id = int(classes_ids[idx])
            color = tuple(get_color(label_id).tolist())
        else:
            color = tuple(get_color(idx).tolist())
        if classes is not None:
            label = classes[idx]

        xmin, ymin, xmax, ymax = boxes_xyxy[idx]
        pt1 = int(xmin), int(ymin)
        pt2 = int(xmax), int(ymax)
        cv2.rectangle(boxes_image, pt1, pt2, color=color, thickness=3)
        info = f"{label}"
        if conf is not None:
            info += f" {conf[idx]:.2f}"

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


def draw_elipsis(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    width: int | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    delta_x = x2 - x1
    delta_y = y2 - y1
    if width is None:
        img_diag = (h**2 + w**2) ** 0.5
        width = int(img_diag // 300)

    kpts_dist = int((delta_x**2 + delta_y**2) ** 0.5)
    if abs(delta_x) > abs(delta_y):
        a = kpts_dist // 2
        b = width
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    else:
        a = width
        b = kpts_dist // 2
        angle = np.arctan2(-delta_x, delta_y) * 180 / np.pi
    image = cv2.ellipse(image, (center_x, center_y), (a, b), angle, 0, 360, color, -1)
    return image


def plot_kpts(
    image: np.ndarray,
    kpts_coords: np.ndarray,
    kpts_scores: np.ndarray | None = None,
    limbs: list[tuple[int, int]] | None = None,
    thr: float = 0.05,
    color_mode: Literal["person", "limb"] = "person",
    alpha: float = 0.8,
) -> np.ndarray:
    """
    kpts_coords is of shape [num_obj, num_kpts, 2]
    grouped_kpts_scores is of shape [num_obj, num_kpts, 1]
    """
    num_obj, num_kpts = kpts_coords.shape[:2]
    if kpts_scores is None:
        kpts_scores = np.ones((num_obj, num_kpts, 1))
    kpts_image = image.copy()
    objs_sizes = kpts_coords[..., 1].max(1) - kpts_coords[..., 1].min(1)
    objs_draw_sizes = (objs_sizes / 100).astype(np.int32)
    for i in range(num_obj):
        obj_draw_size = max(2, objs_draw_sizes[i])
        obj_kpts_coords = kpts_coords[i]
        obj_kpts_scores = kpts_scores[i]
        if color_mode == "person":
            color = get_color(i).tolist()

        # draw limbs connections
        if limbs is not None:
            for j, (idx_0, idx_1) in enumerate(limbs):
                if obj_kpts_scores[idx_0] < thr or obj_kpts_scores[idx_1] < thr:
                    continue
                x1, y1 = obj_kpts_coords[idx_0]
                x2, y2 = obj_kpts_coords[idx_1]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)
                if color_mode == "limb":
                    color = get_color(j).tolist()
                kpts_image = draw_elipsis(kpts_image, x1, y1, x2, y2, color, width=obj_draw_size)
                # kpts_image = cv2.line(kpts_image, (x1, y1), (x2, y2), color, thickness)

        # draw keypoints
        for j, ((x, y), score) in enumerate(zip(obj_kpts_coords, obj_kpts_scores)):
            if score < thr:
                continue
            if color_mode == "limb":
                color = get_color(j).tolist()
            x, y = int(x), int(y)
            cv2.circle(kpts_image, (x, y), obj_draw_size, color, -1)
            cv2.circle(kpts_image, (x, y), obj_draw_size + 1, (0, 0, 0), 1)
    return cv2.addWeighted(image, 1 - alpha, kpts_image, alpha, 0.0)
