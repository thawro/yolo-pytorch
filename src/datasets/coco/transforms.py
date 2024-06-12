import math
import random
from typing import Literal

import cv2
import numpy as np
from torchvision.transforms import functional as F

from src.base.transforms import ComposeTransform
from src.datasets.coco.coco import CocoSample
from src.datasets.coco.utils import segment2box

BORDER_VALUE = (114, 114, 114)


class ComposeCocoTransform(ComposeTransform):
    def __call__(self, sample: CocoSample) -> CocoSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample: CocoSample) -> CocoSample:
        sample.image = F.to_tensor(sample.image)
        sample.transform.norm_scaler = 255
        sample.transform.permute_channels = [1, 2, 0]  # [2, 0, 1]
        return sample


class Normalize:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: CocoSample) -> CocoSample:
        sample.image = F.normalize(sample.image, mean=self.mean, std=self.std)
        sample.transform.norm_mean = self.mean
        sample.transform.norm_std = self.std
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: CocoSample) -> CocoSample:
        if random.random() >= self.p:
            return sample
        sample.horizontal_flip()
        return sample


class RandomAffine:
    height: int
    width: int
    size: tuple[int, int]

    def __init__(
        self,
        degrees: int = 0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.height = -1
        self.width = -1

    def affine_transform(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Center
        x_perspective = random.uniform(-self.perspective, self.perspective)
        y_perspective = random.uniform(-self.perspective, self.perspective)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        x_shear = random.uniform(-self.shear, self.shear)
        y_shear = random.uniform(-self.shear, self.shear)
        x_translation = random.uniform(0.5 - self.translate, 0.5 + self.translate)
        y_translation = random.uniform(0.5 - self.translate, 0.5 + self.translate)

        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -self.width / 2  # x translation (pixels)
        C[1, 2] = -self.height / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = x_perspective  # x perspective (about y)
        P[2, 1] = y_perspective  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(x_shear * math.pi / 180)  # x shear in degrees
        S[1, 0] = math.tan(y_shear * math.pi / 180)  # y shear in degrees

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = x_translation * self.width  # x translation in pixels
        T[1, 2] = y_translation * self.height  # y translation in pixels

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image

        if self.perspective:
            image = cv2.warpPerspective(image, M, dsize=self.size, borderValue=BORDER_VALUE)
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=self.size, borderValue=BORDER_VALUE)
        return image, M, scale

    def apply_boxes(self, boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        num_obj = len(boxes)
        if num_obj == 0:
            return boxes

        xy = np.ones((num_obj * 4, 3), dtype=boxes.dtype)
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            num_obj * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if self.perspective:
            xy = xy[:, :2] / xy[:, 2:3]
        else:
            xy = xy[:, :2]
        xy = xy.reshape(num_obj, 8)

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        return (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=boxes.dtype)
            .reshape(4, num_obj)
            .T
        )

    def apply_segments(self, segments: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply affine to segments and generate new bboxes from segments.

        Args:
            segments (ndarray): list of segments, [num_samples, 500, 2].
            M (ndarray): affine matrix.

        Returns:
            new_segments (ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes after affine, [N, 4].
        """
        num_obj, num_points = segments.shape[:2]
        if num_obj == 0:
            return np.array([]), segments

        xy = np.ones((num_obj * num_points, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(num_obj, -1, 2)
        boxes = np.stack([segment2box(xy, self.height, self.width) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(boxes[:, 0:1], boxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(boxes[:, 1:2], boxes[:, 3:4])
        return boxes, segments

    def apply_keypoints(self, kpts: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine to keypoints.

        Args:
            keypoints (ndarray): keypoints, [N, 17, 3].
            M (ndarray): affine matrix.

        Returns:
            new_keypoints (ndarray): keypoints after affine, [N, 17, 3].
        """
        num_obj, num_kpts = kpts.shape[:2]
        if num_obj == 0:
            return kpts
        xy = np.ones((num_obj * num_kpts, 3), dtype=kpts.dtype)
        visible = kpts[..., 2].reshape(num_obj * num_kpts, 1)
        xy[:, :2] = kpts[..., :2].reshape(num_obj * num_kpts, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (
            (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.width) | (xy[:, 1] > self.height)
        )
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(num_obj, num_kpts, 3)

    @staticmethod
    def filter_box_candidates(
        old_boxes: np.ndarray,
        new_boxes: np.ndarray,
        wh_thr=2,
        ar_thr=100,
        area_thr=0.1,
        eps=1e-16,
    ):
        """
        Compute box candidates based on a set of thresholds. This method compares the characteristics of the boxes
        before and after augmentation to decide whether a box is a candidate for further processing.

        Args:
            wh_thr (float, optional): The width and height threshold in pixels. Default is 2.
            ar_thr (float, optional): The aspect ratio threshold. Default is 100.
            area_thr (float, optional): The area ratio threshold. Default is 0.1.
            eps (float, optional): A small epsilon value to prevent division by zero. Default is 1e-16.

        Returns:
            (numpy.ndarray): A boolean array indicating which boxes are candidates based on the given thresholds.
        """

        w1, h1 = old_boxes[:, 2] - old_boxes[:, 0], old_boxes[:, 3] - old_boxes[:, 1]
        w2, h2 = new_boxes[:, 2] - new_boxes[:, 0], new_boxes[:, 3] - new_boxes[:, 1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        size_mask = (w2 > wh_thr) & (h2 > wh_thr)
        area_mask = w2 * h2 / (w1 * h1 + eps) > area_thr
        ratio_mask = ar < ar_thr
        candidate_mask = size_mask & area_mask & ratio_mask
        return candidate_mask

    def __call__(self, sample: CocoSample) -> CocoSample:
        image = sample.image
        height, width = image.shape[:2]
        self.height, self.width = height, width
        self.size = (width, height)
        boxes_xyxy = sample.boxes_xyxy
        segments = sample.segments
        kpts = sample.kpts

        initial_boxes_xyxy = boxes_xyxy.copy()

        image, M, scale = self.affine_transform(image)
        boxes_xyxy = self.apply_boxes(boxes_xyxy, M)

        # Update boxes segmentation is available
        if segments is not None:
            boxes_xyxy, segments = self.apply_segments(segments, M)
            sample.segments = segments

        if kpts is not None:
            kpts = self.apply_keypoints(kpts, M)
            sample.kpts = kpts

        sample.boxes_xyxy = boxes_xyxy
        sample.image = image

        initial_boxes_xyxy[:, 0::2] *= scale
        initial_boxes_xyxy[:, 1::2] *= scale

        candidate_mask = self.filter_box_candidates(
            old_boxes=initial_boxes_xyxy,
            new_boxes=boxes_xyxy,
            area_thr=0.01 if segments is not None else 0.10,
        )
        sample.filter_by(candidate_mask)
        sample.clip_coords(min_x=0, min_y=0, max_x=width, max_y=height)
        return sample


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self,
        size: int | tuple[int, int] = (640, 640),
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    ):
        """Initialize LetterBox object with specific parameters."""
        self.size = size if isinstance(size, tuple) else (size, size)
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, sample: CocoSample) -> CocoSample:
        """Return updated labels and image with added border."""
        image = sample.image
        shape_old = image.shape[:2]
        shape_new = self.size

        h_old, w_old = shape_old
        h_new, w_new = shape_new

        # Scale ratio (new / old)
        r = min(h_new / h_old, w_new / w_old)
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        scale_w, scale_h = r, r  # width, height ratios
        new_unpad = int(round(w_old * r)), int(round(h_old * r))
        pad_w, pad_h = w_new - new_unpad[0], h_new - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            pad_w, pad_h = np.mod(pad_w, self.stride), np.mod(pad_h, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            pad_w, pad_h = 0.0, 0.0
            new_unpad = (w_new, h_new)
            scale_w, scale_h = w_new / w_old, h_new / h_old  # width, height ratios

        if self.center:
            pad_w /= 2  # divide padding into 2 sides
            pad_h /= 2

        if shape_old[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)) if self.center else 0, int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)) if self.center else 0, int(round(pad_w + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BORDER_VALUE
        )  # add border
        sample.transform.set_pad((top, left, bottom, right))
        sample.transform.set_scale((scale_w, scale_h))
        sample.scale_coords(scale_w, scale_h)
        sample.add_coords(pad_w, pad_h)
        sample.image = image
        return sample
