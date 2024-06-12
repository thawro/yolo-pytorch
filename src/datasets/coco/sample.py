from dataclasses import dataclass, field
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pycocotools
import torch
from torch import Tensor

from src.datasets.coco.utils import (
    COCO_KPT_LIMBS,
    COCO_OBJ_LABELS,
    coco91_to_coco80_labels,
    coco_rle_to_seg,
)
from src.datasets.coco.visualization import plot_segmentation
from src.detection.visualization import plot_boxes
from src.keypoints.visualization import plot_connections
from src.utils.files import load_yaml
from src.utils.image import put_txt


def resample_segmentation(segmentation: np.ndarray, n: int = 1000) -> np.ndarray:
    seg = np.concatenate((segmentation, segmentation[0:1, :]), axis=0)
    x = np.linspace(0, len(seg) - 1, n)
    xp = np.arange(len(seg))
    return (
        np.concatenate([np.interp(x, xp, seg[:, i]) for i in range(2)], dtype=np.float32)
        .reshape(2, -1)
        .T
    )


def reduce_segments(segments: list[np.ndarray], mode: Literal["max", "sum"] = "sum") -> np.ndarray:
    if mode == "sum":  # concat segments
        return np.concatenate(segments)
    elif mode == "max":  # get largest segment
        max_area_segment = max(segments, key=cv2.contourArea)
        return max_area_segment
    else:
        raise ValueError()


@dataclass
class CocoObjectAnnotation:
    area: float
    box_xyxy: list[float] | None
    category_id: int
    id: int
    image_id: int
    iscrowd: int
    segmentation: list[np.ndarray] | np.ndarray | None
    num_keypoints: int | None
    keypoints: list[int] | None

    def __post_init__(self):
        self.has_keypoints = self.num_keypoints is not None and self.keypoints is not None
        self.has_segmentation = self.segmentation is not None
        self.has_boxes = self.box_xyxy is not None
        if self.segmentation is not None:
            segmentation = self.segmentation
            if isinstance(segmentation, dict):  # RLE annotation
                segmentation = coco_rle_to_seg(segmentation)
            segmentation = [np.array(seg).reshape(-1, 2) for seg in segmentation]
            segmentation = reduce_segments(segmentation)
            segmentation = resample_segmentation(segmentation, n=100)
            self.segmentation = segmentation


@dataclass
class CocoMetaInformation:
    height: int | None
    width: int | None
    annot_filepath: str | None
    image_filepath: str | None
    crowd_mask_filepath: str | None


class CocoAnnotation:
    objects: list[CocoObjectAnnotation]
    meta: CocoMetaInformation | None

    def __init__(
        self, objects: list[CocoObjectAnnotation], meta: CocoMetaInformation | None = None
    ):
        self.objects = objects
        self.meta = meta

    @classmethod
    def from_coco_annot(
        cls, annots: list[dict], meta: CocoMetaInformation | None = None
    ) -> "CocoAnnotation":
        objects = []
        for annot in annots:
            category_id = coco91_to_coco80_labels[annot["category_id"] - 1]
            x, y, w, h = annot["bbox"]
            box_xyxy = [x, y, x + w, y + h]
            annot = CocoObjectAnnotation(
                area=annot["area"],
                box_xyxy=box_xyxy,
                category_id=category_id,
                id=annot["id"],
                image_id=annot["image_id"],
                iscrowd=annot["iscrowd"],
                segmentation=annot.get("segmentation", None),
                num_keypoints=annot.get("num_keypoints", None),
                keypoints=annot.get("keypoints", None),
            )
            objects.append(annot)

        return CocoAnnotation(objects, meta)

    @classmethod
    def from_yaml(cls, annot_filepath: str) -> "CocoAnnotation":
        annot = load_yaml(annot_filepath)
        objects = annot["objects"]
        meta = CocoMetaInformation(**annot["meta"])
        return cls.from_coco_annot(objects, meta)

    def get_joints(self) -> np.ndarray:
        num_people = len(self)
        num_kpts = 17
        joints = np.zeros((num_people, num_kpts, 3))
        for i, obj in enumerate(self.objects):
            joints[i] = np.array(obj.keypoints).reshape([-1, 3])
        return joints

    def get_crowd_mask(self, height: int, width: int) -> np.ndarray:
        m = np.zeros((height, width))
        for obj in self.objects:
            rles = []
            if obj.iscrowd:
                rle = pycocotools.mask.frPyObjects(obj.segmentation, height, width)
                rles.append(rle)
            elif obj.has_keypoints and obj.has_segmentation and obj.num_keypoints == 0:
                rle = pycocotools.mask.frPyObjects(obj.segmentation, height, width)
                rles.extend(rle)
            for rle in rles:
                m += pycocotools.mask.decode(rle)
        return m < 0.5

    def __len__(self) -> int:
        return len(self.objects)


@dataclass
class SampleTransform:
    scale_w: float = 1
    scale_h: float = 1
    pad_top: int = 0
    pad_left: int = 0
    pad_bottom: int = 0
    pad_right: int = 0
    norm_scaler: float = 1
    norm_mean: list[float] = field(default_factory=lambda: [0, 0, 0])
    norm_std: list[float] = field(default_factory=lambda: [1, 1, 1])
    permute_channels: list[int] = field(default_factory=lambda: [0, 1, 2])
    horizontal_flip: bool = False

    def set_scale(self, scale_wh: tuple[float, float]):
        self.scale_w = scale_wh[0]
        self.scale_h = scale_wh[1]

    def set_pad(self, pad_tlbr: tuple[int, int, int, int]):
        self.pad_top = pad_tlbr[0]
        self.pad_left = pad_tlbr[1]
        self.pad_bottom = pad_tlbr[2]
        self.pad_right = pad_tlbr[3]

    def get_scale_wh(self) -> tuple[float, float]:
        return self.scale_w, self.scale_h

    def get_pad_tlbr(self) -> tuple[int, int, int, int]:
        return self.pad_top, self.pad_left, self.pad_bottom, self.pad_right

    def reset_preprocessing(self):
        self.permute_channels = [0, 1, 2]
        self.norm_scaler = 1
        self.norm_mean = [0, 0, 0]
        self.norm_std = [1, 1, 1]

    def reset_transforms(self):
        self.set_scale((1, 1))
        self.set_pad((0, 0, 0, 0))
        self.horizontal_flip = False


class CocoSample:
    image: np.ndarray | torch.Tensor
    crowd_mask: np.ndarray | None
    boxes_xyxy: np.ndarray | None
    boxes_cls: np.ndarray | None
    kpts: np.ndarray | None
    segments: np.ndarray | None
    is_crowd_obj = np.ndarray
    transform: SampleTransform

    def __init__(self, image: np.ndarray, annot: CocoAnnotation, crowd_mask: np.ndarray):
        self.image = image
        self.crowd_mask = crowd_mask
        self.has_keypoints = any(obj.has_keypoints for obj in annot.objects)
        self.has_segmentation = any(obj.has_segmentation for obj in annot.objects)
        self.has_boxes = any(obj.has_boxes for obj in annot.objects)
        self.boxes_xyxy, self.kpts, self.segments = None, None, None
        if self.has_boxes:
            self.boxes_xyxy = np.array([obj.box_xyxy for obj in annot.objects])
            self.boxes_cls = np.array([obj.category_id for obj in annot.objects])
        if self.has_keypoints:
            self.kpts = np.array([obj.keypoints for obj in annot.objects])
        if self.has_segmentation:
            self.segments = np.array([obj.segmentation for obj in annot.objects])

        self.is_crowd_obj = np.array(
            [
                obj.iscrowd == 0 and (obj.num_keypoints is None or obj.num_keypoints >= 1)
                for obj in annot.objects
            ]
        )
        self.transform = SampleTransform()

    def scale_coords(self, scale_x: float, scale_y: float):
        if self.boxes_xyxy is not None:
            self.boxes_xyxy[:, 0::2] *= scale_x
            self.boxes_xyxy[:, 1::2] *= scale_y
        if self.segments is not None:
            self.segments[..., 0] *= scale_x
            self.segments[..., 1] *= scale_y
        if self.kpts is not None:
            self.kpts[..., 0] *= scale_x
            self.kpts[..., 1] *= scale_y

    def add_coords(self, add_x: float, add_y: float):
        if self.boxes_xyxy is not None:
            self.boxes_xyxy[:, 0::2] += add_x
            self.boxes_xyxy[:, 1::2] += add_y
        if self.segments is not None:
            self.segments[..., 0] += add_x
            self.segments[..., 1] += add_y
        if self.kpts is not None:
            self.kpts[..., 0] += add_x
            self.kpts[..., 1] += add_y

    def clip_coords(self, min_x: float, min_y: float, max_x: float, max_y: float):
        if self.boxes_xyxy is not None:
            self.boxes_xyxy[:, 0::2] = self.boxes_xyxy[:, 0::2].clip(min_x, max_x)
            self.boxes_xyxy[:, 1::2] = self.boxes_xyxy[:, 1::2].clip(min_y, max_y)
        if self.segments is not None:
            self.segments[..., 0] = self.segments[..., 0].clip(min_x, max_x)
            self.segments[..., 1] = self.segments[..., 1].clip(min_y, max_y)

        if self.kpts is not None:
            self.kpts[..., 0] = self.kpts[..., 0].clip(min_x, max_x)
            self.kpts[..., 1] = self.kpts[..., 1].clip(min_y, max_y)

    def horizontal_flip(self):
        h, w = self.image.shape[:2]
        self.image = self.image[:, ::-1] - np.zeros_like(self.image)
        self.transform.pad_left, self.transform.pad_right = (
            self.transform.pad_right,
            self.transform.pad_left,
        )
        self.transform.horizontal_flip = not self.transform.horizontal_flip

        if self.boxes_xyxy is not None:
            x1 = self.boxes_xyxy[:, 0].copy()
            x2 = self.boxes_xyxy[:, 2].copy()
            self.boxes_xyxy[:, 0] = w - x2
            self.boxes_xyxy[:, 2] = w - x1
        if self.segments is not None:
            self.segments[..., 0] = w - self.segments[..., 0]
        if self.kpts is not None:
            self.kpts[..., 0] = w - self.kpts[..., 0]

    def inverse_preprocessing(self):
        if isinstance(self.image, Tensor):
            self.image = self.image.numpy()
        self.image = self.image.transpose(*self.transform.permute_channels)
        self.image = self.image * self.transform.norm_std + self.transform.norm_mean
        self.image = self.image * self.transform.norm_scaler
        self.transform.reset_preprocessing()

    def inverse_transform(self):
        img_h, img_w = self.image.shape[:2]
        scale_w, scale_h = self.transform.get_scale_wh()
        pad_top, pad_left, pad_bottom, pad_right = self.transform.get_pad_tlbr()
        top, left = pad_top, pad_left
        bottom = img_h - pad_bottom
        right = img_w - pad_right
        self.image = self.image[top:bottom, left:right]
        self.image = cv2.resize(self.image, (0, 0), fx=1 / scale_w, fy=1 / scale_h)
        self.add_coords(-pad_left, -pad_top)
        self.scale_coords(1 / scale_w, 1 / scale_h)
        if self.transform.horizontal_flip:
            self.horizontal_flip()
        self.transform.reset_transforms()

    def filter_by(self, mask: list[bool] | np.ndarray):
        if self.boxes_xyxy is not None:
            self.boxes_xyxy = self.boxes_xyxy[mask]
        if self.boxes_cls is not None:
            self.boxes_cls = self.boxes_cls[mask]
        if self.segments is not None:
            self.segments = self.segments[mask]
        if self.kpts is not None:
            self.kpts = self.kpts[mask]
        self.is_crowd_obj = self.is_crowd_obj[mask]

    def plot(
        self,
        max_size: int | None,
        segmentation: bool = False,
        keypoints: bool = False,
        boxes: bool = False,
        title: str | None = "image",
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        overlay = self.image.copy()
        image_plot = self.image.copy()
        if max_size is None:
            max_size = max(self.image.shape[:2])
        if keypoints and self.kpts is not None:
            kpts = self.kpts[..., :2].astype(np.int32)
            visibility = self.kpts[..., 2]
            image_plot = plot_connections(image_plot, kpts, visibility, COCO_KPT_LIMBS, thr=0.5)

        if segmentation and self.segments is not None:
            image_plot = plot_segmentation(
                image_plot.copy(), self.segments.astype(np.int32), overlay, alpha=0.4
            )

        if boxes and self.boxes_xyxy is not None and self.boxes_cls is not None:
            boxes_labels = [COCO_OBJ_LABELS[label_id] for label_id in self.boxes_cls]
            image_plot = plot_boxes(
                image_plot, self.boxes_xyxy.astype(np.int32), boxes_labels, self.boxes_cls
            )
        raw_h, raw_w = image_plot.shape[:2]

        raw_img_transform = A.Compose(
            [
                A.LongestMaxSize(max_size),
                A.PadIfNeeded(
                    max_size, max_size, border_mode=cv2.BORDER_CONSTANT, value=border_value
                ),
            ],
        )
        image_plot = raw_img_transform(image=image_plot)["image"]
        put_txt(image_plot, [title, f"Shape: {raw_h} x {raw_w}"])
        return image_plot
