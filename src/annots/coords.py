from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from .ops import clip_boxes, xywh2xyxy, xyxy2xywh
from .visuzalization import plot_boxes, plot_kpts, plot_segments

BoxesFormat = Literal["xyxy", "xywh"]
formats = ["xyxy", "xywh"]


class CoordsAnnot(ABC):
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def scale_coords(self, scale_x: float, scale_y: float):
        raise NotImplementedError()

    @abstractmethod
    def add_coords(self, add_x: float, add_y: float):
        raise NotImplementedError()

    @abstractmethod
    def clip_coords(self, min_x: float, min_y: float, max_x: float, max_y: float):
        raise NotImplementedError()

    @abstractmethod
    def horizontal_flip(self, img_w: int, flip_idx: list[int] | None = None):
        raise NotImplementedError()

    @abstractmethod
    def vertical_flip(self, img_h: int, flip_idx: list[int] | None = None):
        raise NotImplementedError()

    def plot(self, scene: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def filter_by(self, mask: list[bool] | np.ndarray):
        self.data = self.data[mask]


class Boxes(CoordsAnnot):
    format: str

    def __init__(self, data: np.ndarray, format: BoxesFormat):
        assert format in formats, f"Invalid box format ({format}), format must be one of {formats}"
        super().__init__(data)
        self.format = format

    @property
    def area(self) -> np.ndarray:
        if self.format == "xyxy":
            return (self.data[:, 2] - self.data[:, 0]) * (self.data[:, 3] - self.data[:, 1])
        return self.data[:, 3] * self.data[:, 2]

    def convert(self, format: BoxesFormat) -> bool:
        """Converts bounding box format from one type to another."""
        is_converted = False
        assert format in formats, f"Invalid box format ({format}), format must be one of {formats}"
        if self.format == format or self.data is None:
            return is_converted
        elif format == "xyxy":
            self.data = xywh2xyxy(self.data)
        elif format == "xywh":
            self.data = xyxy2xywh(self.data)
        self.format = format
        is_converted = True
        return is_converted

    @property
    def is_xyxy(self) -> bool:
        return self.format == "xyxy"

    def scale_coords(self, scale_x: float, scale_y: float):
        self.data[:, 0] *= scale_x
        self.data[:, 1] *= scale_y
        self.data[:, 2] *= scale_x
        self.data[:, 3] *= scale_y

    def add_coords(self, add_x: float, add_y: float):
        self.data[:, 0] += add_x
        self.data[:, 1] += add_y
        if self.is_xyxy:
            self.data[:, 2] += add_x
            self.data[:, 3] += add_y

    def clip_coords(self, min_x: float, min_y: float, max_x: float, max_y: float):
        prev_format = str(self.format)
        is_converted = self.convert("xyxy")
        self.data = clip_boxes(self.data, min_x, min_y, max_x, max_y)
        if is_converted:
            self.convert(prev_format)

    def horizontal_flip(self, img_w: int):
        if self.is_xyxy:
            x1 = self.data[:, 0].copy()
            x2 = self.data[:, 2].copy()
            self.data[:, 0] = img_w - x2
            self.data[:, 2] = img_w - x1
        else:
            self.data[:, 0] = img_w - self.data[:, 0]

    def vertical_flip(self, img_h: int):
        if self.is_xyxy:
            y1 = self.data[:, 1].copy()
            y2 = self.data[:, 3].copy()
            self.data[:, 1] = img_h - y2
            self.data[:, 3] = img_h - y1
        else:
            self.data[:, 1] = img_h - self.data[:, 1]

    def plot(
        self,
        scene: np.ndarray,
        classes: np.ndarray | list[str] | None = None,
        classes_ids: np.ndarray | list[int] | None = None,
        conf: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        if not self.is_xyxy:
            prev_format = str(self.format)
            self.convert("xyxy")
            boxes_xyxy = self.data.copy()
            self.convert(prev_format)
        else:
            boxes_xyxy = self.data.copy()
        return plot_boxes(scene, boxes_xyxy, classes, classes_ids, conf)


class Keypoints(CoordsAnnot):
    def scale_coords(self, scale_x: float, scale_y: float):
        self.data[..., 0] *= scale_x
        self.data[..., 1] *= scale_y

    def add_coords(self, add_x: float, add_y: float):
        self.data[..., 0] += add_x
        self.data[..., 1] += add_y

    def clip_coords(self, min_x: float, min_y: float, max_x: float, max_y: float):
        self.data[..., 0] = self.data[..., 0].clip(min_x, max_x)
        self.data[..., 1] = self.data[..., 1].clip(min_y, max_y)

    def horizontal_flip(self, img_w: int, flip_idx: list[int] | None = None):
        self.data[..., 0] = img_w - self.data[..., 0]
        if flip_idx is not None:
            self.img_h = np.ascontiguousarray(self.img_h[:, flip_idx])

    def vertical_flip(self, img_h: int, flip_idx: list[int] | None = None):
        self.data[..., 1] = img_h - self.data[..., 1]
        if flip_idx is not None:
            self.img_h = np.ascontiguousarray(self.img_h[:, flip_idx])

    def plot(self, scene: np.ndarray, limbs: list[tuple[int, int]] | None = None) -> np.ndarray:
        return plot_kpts(scene, self.data[..., :2].astype(np.int32), limbs=limbs)


class Segments(Keypoints):
    def horizontal_flip(self, img_w: int):
        super().horizontal_flip(img_w, flip_idx=None)

    def vertical_flip(self, img_h: int):
        super().vertical_flip(img_h, flip_idx=None)

    def plot(self, scene: np.ndarray) -> np.ndarray:
        return plot_segments(scene, self.data.astype(np.int32), alpha=0.4)
