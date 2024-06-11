import random
from pathlib import Path

import cv2
import numpy as np
import pycocotools
import torch
from PIL import Image
from torch import Tensor

from src.base.transforms import inverse_preprocessing
from src.datasets.coco import CocoDataset
from src.keypoints.transforms import ComposeKeypointsTransform, KeypointsTransform
from src.keypoints.visualization import plot_connections, plot_heatmaps
from src.logger.pylogger import log
from src.utils.image import get_color, make_grid, put_txt, stack_horizontally


def get_coco_joints(annots: list[dict]):
    num_people = len(annots)
    num_kpts = 17
    joints = np.zeros((num_people, num_kpts, 3))
    for i, obj in enumerate(annots):
        joints[i] = np.array(obj["keypoints"]).reshape([-1, 3])
    return joints


class HeatmapGenerator:
    """
    source: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py
    """

    def __init__(self, num_kpts: int, size: int, sigma: float = 2):
        self.num_kpts = num_kpts
        self.size = size
        self.h, self.w = size, size
        if sigma < 0:
            sigma = size / 64
        self.sigma = sigma
        x = np.arange(0, 6 * sigma + 3, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(
        self,
        joints: np.ndarray,
        # max_h: int,
        # max_w: int,
    ) -> np.ndarray:
        """
        visibility: 0 - not labeled, 1 - labeled but not visible, 2 - labeled and visible
        """
        hms = np.zeros((self.num_kpts, self.h, self.w), dtype=np.float32)
        for joint in joints:
            for idx in range(self.num_kpts):
                x, y, vis = joint[idx]
                if vis <= 0 or x < 0 or y < 0 or x >= self.w or y >= self.h:
                    continue

                xmin = int(np.round(x - 3 * self.sigma - 1))
                ymin = int(np.round(y - 3 * self.sigma - 1))
                xmax = int(np.round(x + 3 * self.sigma + 2))
                ymax = int(np.round(y + 3 * self.sigma + 2))

                c, d = max(0, -xmin), min(xmax, self.w) - xmin
                a, b = max(0, -ymin), min(ymax, self.h) - ymin

                cc, dd = max(0, xmin), min(xmax, self.w)
                aa, bb = max(0, ymin), min(ymax, self.h)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.gauss[a:b, c:d])
        return hms


class JointsGenerator:
    def __init__(self, size: int = 512):
        self.h, self.w = size, size

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        for i in range(len(joints)):
            for k, pt in enumerate(joints[i]):
                x, y, vis = int(pt[0]), int(pt[1]), pt[2]
                if vis > 0 and x >= 0 and y >= 0 and x < self.w and y < self.h:
                    joints[i, k] = (x, y, 1)
                else:
                    joints[i, k] = (0, 0, 0)
        visible_joints_mask = joints.sum(axis=(1, 2)) > 0
        return joints[visible_joints_mask].astype(np.int32)


def collate_fn(
    batch: list[tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]],
) -> tuple[
    Tensor,
    list[Tensor],
    list[Tensor],
    list[list[np.ndarray]],
]:
    num_scales = len(batch[0][1])
    images = torch.from_numpy(np.stack([item[0] for item in batch]))
    heatmaps_tensor = []
    masks_tensor = []
    joints_scales = []
    for i in range(num_scales):
        _heatmaps = []
        _masks = []
        _joints = []
        for sample in batch:
            _heatmaps.append(sample[1][i])
            _masks.append(sample[2][i])
            _joints.append(sample[3][i])
        heatmaps_tensor.append(torch.from_numpy(np.stack(_heatmaps)))
        masks_tensor.append(torch.from_numpy(np.stack(_masks)))
        joints_scales.append(_joints)
    return images, heatmaps_tensor, masks_tensor, joints_scales


def get_crowd_mask(annot: list, img_h: int, img_w: int) -> np.ndarray:
    m = np.zeros((img_h, img_w))
    for obj in annot:
        if obj["iscrowd"]:
            rle = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            m += pycocotools.mask.decode(rle)
        elif obj["num_keypoints"] == 0:
            rles = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            for rle in rles:
                m += pycocotools.mask.decode(rle)
    return m < 0.5


class CocoKeypointsDataset(CocoDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: ComposeKeypointsTransform | None = None,
        out_size: int = 512,
        hm_resolutions: list[float] = [1 / 4, 1 / 2],
        num_kpts: int = 17,
        max_num_people: int = 30,
        sigma: float = 2,
        mosaic_probability: float = 0,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            out_size=out_size,
            mosaic_probability=mosaic_probability,
        )
        self.num_scales = len(hm_resolutions)
        self.num_kpts = num_kpts
        self.max_num_people = max_num_people
        self.hm_resolutions = hm_resolutions
        self.hm_sizes = [int(res * out_size) for res in hm_resolutions]
        hm_generators = []
        joints_generator = []
        for hm_size in self.hm_sizes:
            hm_generators.append(HeatmapGenerator(self.num_kpts, hm_size, sigma=sigma))
            joints_generator.append(JointsGenerator(hm_size))
        self.hm_generators = hm_generators
        self.joints_generators = joints_generator

    def plot_examples(
        self, idxs: list[int], nrows: int = 1, stage_idxs: list[int] = [1, 0]
    ) -> np.ndarray:
        return super().plot_examples(idxs, nrows=nrows, stage_idxs=stage_idxs)

    def plot(self, idx: int, nrows: int = 3, stage_idxs: list[int] = [1]) -> np.ndarray:
        def plot_stage(stage_idx: int) -> np.ndarray:
            crowd_mask = mask_list[stage_idx]
            h, w = crowd_mask.shape[:2]
            filename = Path(self.images_filepaths[idx]).stem
            if stage_idx == 1:
                font_scale = 0.5
                labels = [
                    "Transformed Image",
                    f"Sample: {filename}",
                    f"Stage: {stage_idx} ({h} x {w})",
                ]
            else:
                font_scale = 0.25
                labels = [f"Stage: {stage_idx} ({h} x {w})"]
            image = cv2.resize(img_npy, (w, h))
            kpts_coords = joints_list[stage_idx][..., :2]
            visibility = joints_list[stage_idx][..., 2]
            kpts_heatmaps = plot_heatmaps(image, heatmaps[stage_idx])
            image = plot_connections(
                image.copy(), kpts_coords, visibility, self.kpts_limbs, thr=0.5
            )
            crowd_mask = cv2.cvtColor((crowd_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            if stage_idx == 1:
                for i in range(len(kpts_heatmaps)):
                    put_txt(
                        kpts_heatmaps[i], [self.kpts_limbs[i]], alpha=0.8, font_scale=font_scale
                    )
                put_txt(crowd_mask, ["Crowd Mask"], alpha=1, font_scale=font_scale)
            put_txt(image, labels, alpha=0.8, font_scale=font_scale)
            grid = make_grid([image, crowd_mask, *kpts_heatmaps], nrows=nrows).astype(np.uint8)
            return grid

        img, heatmaps, mask_list, joints_list = self[idx]
        img_npy = inverse_preprocessing(img)
        stages_grids = [plot_stage(stage_idx) for stage_idx in stage_idxs]
        model_input_grid = make_grid(stages_grids, nrows=len(stages_grids), match_size=True).astype(
            np.uint8
        )
        raw_image = self.plot_raw(idx, max_size=model_input_grid.shape[0])
        sample_vis = stack_horizontally([raw_image, model_input_grid])
        return sample_vis

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ]:
        if random.random() < self.mosaic_probability:
            img, annot, mask = self.get_raw_mosaiced_data(idx, use_keypoints=True)
        else:
            img, annot, mask = self.get_raw_data_with_crowd_mask(idx)

        annots = [obj for obj in annot if obj["iscrowd"] == 0 or obj["num_keypoints"] > 0]
        joints = get_coco_joints(annots)
        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        if self.transform is not None:
            img, mask_list, joints_list = self.transform(img, mask_list, joints_list)
        heatmaps = []
        for i in range(self.num_scales):
            joints_list[i] = self.joints_generators[i](joints_list[i])
            hms = self.hm_generators[i](joints_list[i])
            heatmaps.append(hms.astype(np.float32))
        return img, heatmaps, mask_list, joints_list


_polygons = list[list[int]]


k_i = [26, 25, 25, 35, 35, 79, 79, 72, 72, 62, 62, 107, 107, 87, 87, 89, 89]
k_i = np.array(k_i) / 1000
variances = (k_i * 2) ** 2


def object_OKS(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    obj_polygons: _polygons,
) -> float:
    # pred_kpts shape: [num_kpts, 2]
    # target_kpts shape: [num_kpts, 2]
    # 2 for: x, y
    if target_vis.sum() <= 0:
        return -1

    kpts_vis = target_vis > 0
    area = sum(
        cv2.contourArea(np.array(poly).reshape(-1, 2).astype(np.int32)) for poly in obj_polygons
    )
    area += np.spacing(1)

    dist = ((pred_kpts - target_kpts) ** 2).sum(-1)
    # dist is already squared (euclidean distance has square root)
    e = dist / (2 * variances * area)
    e = e[kpts_vis]
    num_vis_kpts = kpts_vis.sum()
    e = np.exp(-e)
    oks = np.sum(e) / num_vis_kpts
    return oks


def image_OKS(
    pred_kpts: np.ndarray,
    target_kpts: np.ndarray,
    target_vis: np.ndarray,
    seg_polygons: list[_polygons],
) -> float:
    num_obj = len(target_kpts)
    oks_values = []
    for j in range(num_obj):
        dist = ((pred_kpts[j] - target_kpts[j]) ** 2).sum(-1) ** 0.5
        oks = object_OKS(pred_kpts[j], target_kpts[j], target_vis[j], seg_polygons[j])
        oks_values.append(oks)

    oks_values = np.array(oks_values).round(3)
    valid_oks_mask = oks_values != -1
    if valid_oks_mask.sum() > 0:
        oks_avg = oks_values[valid_oks_mask].mean()
        return oks_avg
    return -1


if __name__ == "__main__":
    from PIL import Image

    from src.keypoints.transforms import KeypointsTransform

    out_size = 512
    hm_resolutions = [1 / 4, 1 / 2]
    transform = KeypointsTransform(out_size, hm_resolutions, min_scale=0.25, max_scale=1.25)
    # ds = CocoKeypointsDataset("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = CocoKeypointsDataset(
        "data/COCO/raw",
        "train2017",
        transform.train,
        out_size,
        hm_resolutions,
        mosaic_probability=0.25,
    )
    grid = ds.plot_examples([0, 1, 2, 3, 4, 5, 6], nrows=1, stage_idxs=[1, 0])
    Image.fromarray(grid).save("test.jpg")
    # ds.explore(0)
