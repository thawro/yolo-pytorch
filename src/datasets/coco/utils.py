import cv2
import numpy as np
import torch
from torch import Tensor

_xy_coords = tuple[int, int]
_polygon = list[_xy_coords]
_polygons = list[_polygon]

# list of smaller polygons [[x11, y11, x12, y12, ..., x1n, y1n], [x21, y21, ...]]
_coco_polygon = list[float]
_coco_polygons = list[_coco_polygon]
_mask = np.ndarray

COCO_KPT_LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_KPT_LIMBS = [
    (9, 7),
    (7, 5),
    (5, 3),
    (3, 1),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

COCO_OBJ_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

coco91_to_coco80_labels = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    None,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    None,
    24,
    25,
    None,
    None,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    None,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    None,
    60,
    None,
    None,
    61,
    None,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    None,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    None,
]


def mask_to_polygons(mask: _mask) -> _polygons:
    """Return list of polygons. Each polygon is a list of corner coords"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt.squeeze().flatten().tolist() for cnt in contours]
    return contours


def coco_polygons_to_mask(polygons: _coco_polygons, h: int, w: int) -> _mask:
    if isinstance(polygons, dict):
        return coco_rle_seg_to_mask(polygons)
    seg_polygons = [np.array(polygon).reshape(-1, 2).astype(np.int32) for polygon in polygons]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.drawContours(mask, seg_polygons, -1, 255, -1)
    return mask


def coco_rle_seg_to_mask(segmentation: dict[str, list[int]]) -> _mask:
    h, w = segmentation["size"]
    counts = segmentation["counts"]
    values = []
    value = 0
    for count in counts:
        values.extend([value] * count)
        value = (not value) * 255
    mask = np.array(values, dtype=np.uint8).reshape(w, h).T
    return mask


def coco_rle_to_seg(segmentation: dict[str, list[int]]) -> _coco_polygons | _polygons:
    mask = coco_rle_seg_to_mask(segmentation)
    return mask_to_polygons(mask)


# def segment2box(segment: np.ndarray, height: int, width: int) -> np.ndarray:
#     x, y = segment.T  # segment xy
#     inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
#     x = x[inside]
#     y = y[inside]
#     return (
#         np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
#         if any(x)
#         else np.zeros(4, dtype=segment.dtype)
#     )  # xyxy


# def _empty_like(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
#     if isinstance(x, np.ndarray):
#         return np.empty_like(x)
#     else:
#         return torch.empty_like(x)


# def xyxy2xywh(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
#     boxes_ = _empty_like(boxes)
#     xmin, ymin = boxes[..., 0], boxes[..., 1]
#     xmax, ymax = boxes[..., 2], boxes[..., 3]
#     boxes_[..., 0] = (xmin + xmax) / 2  # xc
#     boxes_[..., 1] = (ymin + ymax) / 2  # yc
#     boxes_[..., 2] = xmax - xmin  # w
#     boxes_[..., 3] = ymax - ymin  # h
#     return boxes_


# def xywh2xyxy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
#     boxes_ = _empty_like(boxes)
#     xy = boxes[..., :2]  # centers
#     wh = boxes[..., 2:] / 2  # half wh
#     boxes_[..., :2] = xy - wh  # xmin, ymin
#     boxes_[..., 2:] = xy + wh  # xmax, ymax
#     return boxes_
