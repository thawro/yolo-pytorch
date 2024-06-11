from typing import Callable

import cv2
import numpy as np
from torch import Tensor

from src.utils.utils import _norm, list2array

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def inverse_preprocessing(
    image: Tensor | np.ndarray, mean: _norm = MEAN, std: _norm = STD
) -> np.ndarray:
    if isinstance(image, Tensor):
        image_npy = image.detach().cpu().numpy()
    image_npy = image_npy.transpose(1, 2, 0)
    image_npy = (image_npy * list2array(std)) + list2array(mean)
    image_npy = (image_npy * 255).astype(np.uint8)
    return image_npy


def affine_transform(point: tuple[int, int], transform_matrix: np.ndarray):
    new_pt = np.array([point[0], point[1], 1.0]).T
    new_pt = np.dot(transform_matrix, new_pt)
    return new_pt[:2]


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point: tuple[float, float], rot_rad: float) -> tuple[float, float]:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = (
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    )
    return src_result


def get_affine_transform(
    center: tuple[int, int],
    scale: tuple[float, float],
    rot: float,
    output_size: tuple[int, int],
    shift: tuple[int, int] = (0, 0),
    inverse: bool = False,
) -> np.ndarray:
    shift = np.array(shift)
    scale = np.array(scale)
    center = np.array(center)

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, -src_w / 2], rot_rad)
    dst_dir = np.array([0, -dst_w / 2], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inverse:
        src, dst = dst, src
    return cv2.getAffineTransform(src, dst)


def get_multi_scale_size(
    image: np.ndarray,
    input_size: int,
    current_scale: float,
    min_scale: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
    h, w, _ = image.shape
    center = (int(w / 2.0 + 0.5), int(h / 2.0 + 0.5))

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63) // 64 * 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(int((min_input_size / w * h + 63) // 64 * 64) * current_scale / min_scale)
        scale_w = w
        scale_h = h_resized / w_resized * w
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(int((min_input_size / h * w + 63) // 64 * 64) * current_scale / min_scale)
        scale_h = h
        scale_w = w_resized / h_resized * h

    return (w_resized, h_resized), center, (scale_w, scale_h)


def resize_align_multi_scale(
    image: np.ndarray, input_size: int, current_scale: float, min_scale: float
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    size_resized, center, scale = get_multi_scale_size(image, input_size, current_scale, min_scale)
    trans = get_affine_transform(center, scale, 0, size_resized)
    image_resized = cv2.warpAffine(image, trans, size_resized)
    return image_resized, center, scale


class ComposeTransform:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, *args) -> tuple:
        for transform in self.transforms:
            args = transform(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
