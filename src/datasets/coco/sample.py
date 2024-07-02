from dataclasses import dataclass, field
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pycocotools

from src.annots import Boxes, Keypoints, Segments
from src.datasets.coco.utils import (
    COCO_KPT_LIMBS,
    COCO_OBJ_LABELS,
    coco91_to_coco80_labels,
    coco_rle_to_seg,
)
from src.utils.files import load_yaml
from src.utils.image import put_txt

_array_or_none = np.ndarray | None


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
    segmentation: list[np.ndarray] | _array_or_none
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
    def from_dict(cls, annot: dict) -> "CocoAnnotation":
        objects = annot["objects"]
        meta = CocoMetaInformation(**annot["meta"])
        return cls.from_coco_annot(objects, meta)

    @classmethod
    def from_yaml(cls, annot_filepath: str) -> "CocoAnnotation":
        annot = load_yaml(annot_filepath)
        return cls.from_dict(annot)

    def get_joints(self) -> np.ndarray:
        num_people = len(self)
        num_kpts = 17
        joints = np.zeros((num_people, num_kpts, 3))
        for i, obj in enumerate(self.objects):
            joints[i] = np.array(obj.keypoints).reshape([-1, 3])
        return joints

    def get_crowd_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width))
        for obj in self.objects:
            rles = []
            if obj.iscrowd:
                rle = pycocotools.mask.frPyObjects(obj.segmentation, height, width)
                rles.append(rle)
            elif obj.has_keypoints and obj.has_segmentation and obj.num_keypoints == 0:
                rle = pycocotools.mask.frPyObjects(obj.segmentation, height, width)
                rles.extend(rle)
            for rle in rles:
                mask += pycocotools.mask.decode(rle)
        return mask < 0.5

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
    vertical_flip: bool = False

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
        self.vertical_flip = False


class CocoSample:
    image: np.ndarray
    image_filepath: str
    classes: np.ndarray
    is_crowd_obj = np.ndarray
    crowd_mask: np.ndarray
    boxes: Boxes
    segments: Segments
    kpts: Keypoints
    scale_wh: tuple[float, float]
    pad_tlbr: tuple[float, float, float, float]

    def __init__(
        self,
        image: np.ndarray,
        boxes_xyxy: np.ndarray,
        classes: np.ndarray,
        kpts: np.ndarray,
        segments: np.ndarray,
        crowd_mask: np.ndarray,
        is_crowd_obj: np.ndarray,
        image_filepath: str,
    ):
        self.image = image
        self.crowd_mask = crowd_mask
        self.boxes = Boxes(boxes_xyxy, format="xyxy")
        self.kpts = Keypoints(kpts)
        self.segments = Segments(segments)
        self.classes = classes
        self.is_crowd_obj = is_crowd_obj
        self.image_filepath = image_filepath
        self.scale_wh = 1, 1
        self.pad_tlbr = 0, 0, 0, 0

    @property
    def num_obj(self) -> int:
        return len(self.classes)

    def to_batch(self) -> dict[str, np.ndarray]:
        # TODO: add protocol
        batch_sample = {
            "images": self.image,
            "boxes_xywh": self.boxes.data,
            "classes": self.classes,
            "num_obj": self.num_obj,
            "pad_tlbr": self.pad_tlbr,
            "scale_wh": self.scale_wh,
            "image_filepath": self.image_filepath,
        }
        if self.segments.data is not None:
            batch_sample["segments"] = self.segments.data
        if self.kpts.data is not None:
            batch_sample["kpts"] = self.kpts.data
        return batch_sample

    def remove_zero_area_boxes(self):
        valid_area_mask = self.boxes.area > 0
        self.filter_by(valid_area_mask)

    def convert_boxes(self, format: Literal["xyxy", "xywh"]):
        self.boxes.convert(format)

    @property
    def has_boxes(self) -> bool:
        return self.boxes.data is not None

    @property
    def has_segments(self) -> bool:
        return self.segments.data is not None

    @property
    def has_kpts(self) -> bool:
        return self.kpts.data is not None

    def scale_coords(self, scale_x: float, scale_y: float):
        if self.has_boxes:
            self.boxes.scale_coords(scale_x, scale_y)
        if self.has_kpts:
            self.kpts.scale_coords(scale_x, scale_y)
        if self.has_segments:
            self.segments.scale_coords(scale_x, scale_y)

    def add_coords(self, add_x: float, add_y: float):
        if self.has_boxes:
            self.boxes.add_coords(add_x, add_y)
        if self.has_segments:
            self.segments.add_coords(add_x, add_y)
        if self.has_kpts:
            self.kpts.add_coords(add_x, add_y)

    def clip_coords(self, min_x: float, min_y: float, max_x: float, max_y: float):
        if self.has_boxes:
            self.boxes.clip_coords(min_x, min_y, max_x, max_y)
        if self.has_segments:
            self.segments.clip_coords(min_x, min_y, max_x, max_y)
        if self.has_kpts:
            self.kpts.clip_coords(min_x, min_y, max_x, max_y)

    def horizontal_flip(self, flip_idx: list[int] | None = None):
        img_w = self.image.shape[1]
        self.image = np.ascontiguousarray(self.image[:, ::-1])
        if self.crowd_mask is not None:
            self.crowd_mask = np.ascontiguousarray(self.crowd_mask[:, ::-1])

        if self.has_boxes:
            self.boxes.horizontal_flip(img_w)
        if self.has_segments:
            self.segments.horizontal_flip(img_w)
        if self.has_kpts:
            self.kpts.horizontal_flip(img_w, flip_idx)

    def vertical_flip(self, flip_idx: list[int] | None = None):
        img_h = self.image.shape[0]
        self.image = np.ascontiguousarray(self.image[::-1])
        if self.crowd_mask is not None:
            self.crowd_mask = np.ascontiguousarray(self.crowd_mask[::-1])

        if self.has_boxes:
            self.boxes.vertical_flip(img_h)
        if self.has_segments:
            self.segments.vertical_flip(img_h)
        if self.has_kpts:
            self.kpts.vertical_flip(img_h, flip_idx)

    def shapes_info(self):
        names = ["image", "crowd_mask", "boxes", "classes", "kpts", "segments"]
        max_len = max(len(name) for name in names) + 2
        for name in names:
            attr: np.ndarray | None = getattr(self, name, None)
            name = name + ":"
            if attr is not None:
                print(f"{name:<{max_len}}{str(list(attr.shape)):<{20}}{attr.dtype}")
        print()

    def filter_by(self, mask: list[bool] | np.ndarray):
        if self.has_boxes:
            self.boxes.filter_by(mask)
        if self.has_segments:
            self.segments.filter_by(mask)
        if self.has_kpts:
            self.kpts.filter_by(mask)
        if self.classes is not None:
            self.classes = self.classes[mask]
        self.is_crowd_obj = self.is_crowd_obj[mask]

    def plot(
        self,
        max_size: int | None,
        segmentation: bool = False,
        keypoints: bool = False,
        boxes: bool = False,
        title: str = "image",
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        image_plot = self.image.copy()
        if max_size is None:
            max_size = max(self.image.shape[:2])
        if keypoints and self.has_kpts:
            image_plot = self.kpts.plot(image_plot, COCO_KPT_LIMBS)

        if segmentation and self.has_segments:
            image_plot = self.segments.plot(image_plot.copy())

        if boxes and self.has_boxes and self.classes is not None:
            classes = [COCO_OBJ_LABELS[class_id] for class_id in self.classes]
            image_plot = self.boxes.plot(image_plot, classes)
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

    @classmethod
    def from_annot(cls, image: np.ndarray, annot: CocoAnnotation, crowd_mask: np.ndarray):
        boxes, kpts, segments = None, None, None
        has_keypoints = any(obj.has_keypoints for obj in annot.objects)
        has_segmentation = any(obj.has_segmentation for obj in annot.objects)
        has_boxes = any(obj.has_boxes for obj in annot.objects)
        if has_boxes:
            boxes = np.array([obj.box_xyxy for obj in annot.objects])
            classes = np.array([obj.category_id for obj in annot.objects])
        if has_keypoints:
            kpts = np.array([obj.keypoints for obj in annot.objects])
        if has_segmentation:
            segments = np.array([obj.segmentation for obj in annot.objects])

        is_crowd_obj = np.array(
            [
                obj.iscrowd == 0 and (obj.num_keypoints is None or obj.num_keypoints >= 1)
                for obj in annot.objects
            ]
        )
        return CocoSample(
            image=image,
            boxes_xyxy=boxes,
            classes=classes,
            kpts=kpts,
            segments=segments,
            crowd_mask=crowd_mask,
            is_crowd_obj=is_crowd_obj,
        )
