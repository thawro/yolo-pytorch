import glob
import os
import random
from pathlib import Path
from typing import Callable, Literal

import albumentations as A
import cv2
import numpy as np
import pycocotools
import torch
from pycocotools.coco import COCO
from torch import Tensor
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.datasets.coco.utils import (
    COCO_KPT_LABELS,
    COCO_KPT_LIMBS,
    COCO_OBJ_LABELS,
    coco91_to_coco80_labels,
    coco_polygons_to_mask,
    coco_rle_to_seg,
    mask_to_polygons,
)
from src.keypoints.visualization import plot_connections, plot_heatmaps
from src.logger.pylogger import log
from src.utils.files import load_yaml, save_yaml
from src.utils.image import get_color, make_grid, put_txt, stack_horizontally
from src.utils.utils import get_rank

_tasks = Literal["instances", "person_keypoints"]


def get_crowd_mask(annot: list, img_h: int, img_w: int) -> np.ndarray:
    m = np.zeros((img_h, img_w))
    for obj in annot:
        rles = []
        if obj["iscrowd"]:
            rle = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            rles.append(rle)
        elif "num_keypoints" in obj and obj["num_keypoints"] == 0:
            rle = pycocotools.mask.frPyObjects(obj["segmentation"], img_h, img_w)
            rles.extend(rle)
        for rle in rles:
            m += pycocotools.mask.decode(rle)
    return m < 0.5


class CocoDataset(BaseImageDataset):
    name: str = "COCO"
    task: _tasks
    kpts_limbs = COCO_KPT_LIMBS
    kpts_labels = COCO_KPT_LABELS
    objs_labels = COCO_OBJ_LABELS

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable | None = None,
        out_size: int = 512,
        mosaic_probability: float = 0,
    ):
        self.root = root
        self.split = split
        self.is_train = "train" in split
        self.transform = transform
        self.out_size = out_size
        self.mosaic_probability = mosaic_probability
        self.task_split = f"{self.task}_{self.split}"

        self.images_dir = f"{self.root}/images/{self.split}"
        self.annots_dir = f"{self.root}/annotations/{self.task_split}"
        self.crowd_masks_dir = f"{self.root}/crowd_masks/{self.task_split}"
        self._set_paths()

    def _set_paths(self):
        annots_filepaths = glob.glob(f"{self.annots_dir}/*")
        images_filepaths = [
            path.replace(".yaml", ".jpg").replace(
                f"annotations/{self.task_split}", f"images/{self.split}"
            )
            for path in annots_filepaths
        ]
        crowd_masks_filepaths = [
            path.replace(".yaml", ".npy").replace("annotations/", "masks/")
            for path in annots_filepaths
        ]
        self.annots_filepaths = np.array(annots_filepaths, dtype=np.str_)
        self.images_filepaths = np.array(images_filepaths, dtype=np.str_)
        self.crowd_masks_filepaths = np.array(crowd_masks_filepaths, dtype=np.str_)
        num_imgs = len(images_filepaths)
        num_annots = len(annots_filepaths)
        num_crowd_masks = len(crowd_masks_filepaths)
        assert num_imgs == num_annots == num_crowd_masks, (
            f"There must be the same number of images and annotations. "
            f"Currently: num_imgs={num_imgs}, num_annots={num_annots}, num_masks={num_crowd_masks}. "
            "Make sure that `_save_annots_to_files` was already called (run `make save_coco_annots` command in terminal)"
        )

    def save_annots_to_files(self):
        # save crowd_masks to npy files and annots to yaml files
        rank = get_rank()
        if rank != 0:
            log.warn(
                f"     Current process (rank = {rank}) is not the main process (rank = 0) -> Skipping annots files saving"
            )
            return
        annot_dir_exists = os.path.exists(self.annots_dir)
        num_files = len(glob.glob(f"{self.annots_dir}/*"))

        coco = COCO(f"{self.root}/annotations/{self.task_split}.json")
        ids = list(coco.imgs.keys())
        # remove_images_without_annotations
        n_before = len(ids)
        log.info("Removing data samples without annotations")
        ids = [_id for _id in ids if len(coco.getAnnIds(imgIds=_id, iscrowd=None)) > 0]
        log.info(f"before = {n_before}, after = {len(ids)}")
        num_gt_annots = len(ids)
        if annot_dir_exists and num_files == num_gt_annots:
            log.info(
                f"..{self.task_split} annotations already saved to files -> Skipping annots files saving.."
            )
            return
        if num_files != num_gt_annots and annot_dir_exists:
            log.info(
                f"..Number of ground truth annotations ({num_gt_annots}) is not equal to "
                f"number of saves annotation files ({num_files}). "
                f"Repeating the annotations files saving process.."
            )
        log.info(
            f"..Saving {self.task_split} annotations to files (annots to .yaml and crowd masks to .npy).."
        )
        Path(self.annots_dir).mkdir(exist_ok=True, parents=True)
        Path(self.crowd_masks_dir).mkdir(exist_ok=True, parents=True)

        for idx in tqdm(range(len(ids)), desc=f"Saving {self.task_split} annotations"):
            img_id = ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annot = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            img_filepath = f"{self.root}/images/{self.split}/{img_info['file_name']}"
            filename = Path(img_filepath).stem
            annot_filepath = f"{self.annots_dir}/{filename}.yaml"
            mask_filepath = f"{self.crowd_masks_dir}/{filename}.npy"
            mask = get_crowd_mask(annot, img_info["height"], img_info["width"])
            np.save(mask_filepath, mask)
            save_yaml(annot, annot_filepath)

    def load_annot(self, idx: int) -> list[dict]:
        annot = load_yaml(self.annots_filepaths[idx])
        for idx in range(len(annot)):
            annot[idx]["category_id"] = coco91_to_coco80_labels[annot[idx]["category_id"] - 1]
        return annot

    def load_crowd_mask(self, idx: int) -> np.ndarray:
        return np.load(self.crowd_masks_filepaths[idx])

    def get_raw_data(self, idx: int) -> tuple[np.ndarray, list[dict]]:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        return image, annot

    def get_raw_data_with_crowd_mask(self, idx: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
        image, annot = self.get_raw_data(idx)
        mask = np.load(self.crowd_masks_filepaths[idx])
        return image, annot, mask

    def plot_raw(
        self,
        idx: int,
        max_size: int,
        use_segmentation: bool = False,
        use_keypoints: bool = False,
    ) -> np.ndarray:
        raw_image, raw_annot = self.get_raw_data(idx)
        if use_keypoints:
            raw_joints = get_coco_joints(raw_annot)
            kpts = raw_joints[..., :2]
            visibility = raw_joints[..., 2]
            raw_image = plot_connections(raw_image.copy(), kpts, visibility, self.limbs, thr=0.5)
            overlay = raw_image.copy()

        if use_segmentation:
            for i, obj in enumerate(raw_annot):
                seg = obj["segmentation"]
                if isinstance(seg, dict):  # RLE annotation
                    seg = coco_rle_to_seg(seg)
                c = get_color(i).tolist()
                for poly_seg in seg:
                    poly_seg = np.array(poly_seg).reshape(-1, 2).astype(np.int32)
                    overlay = cv2.fillPoly(overlay, [poly_seg], color=c)
                    overlay = cv2.drawContours(overlay, [poly_seg], -1, color=c, thickness=1)

            alpha = 0.4  # Transparency factor.
            raw_image = cv2.addWeighted(overlay, alpha, raw_image, 1 - alpha, 0)

        raw_h, raw_w = raw_image.shape[:2]

        raw_img_transform = A.Compose(
            [
                A.LongestMaxSize(max_size),
                A.PadIfNeeded(
                    max_size, max_size, border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128)
                ),
            ],
        )
        raw_image = raw_img_transform(image=raw_image)["image"]
        put_txt(raw_image, ["Raw Image", f"Shape: {raw_h} x {raw_w}"])
        return raw_image

    def get_raw_mosaiced_data(
        self,
        idx: int,
        use_segmentation: bool = False,
        use_keypoints: bool = False,
        use_crowd_mask: bool = False,
    ) -> tuple[np.ndarray, list[dict], np.ndarray]:
        out_size = self.out_size * 2
        img_size = out_size // 2
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        mosaic_annot = []
        mosaic_img = np.zeros([out_size, out_size, 3], dtype=np.uint8)
        mosaic_mask = np.empty([out_size, out_size], dtype=np.bool_)

        new_h, new_w = img_size, img_size

        for i in range(4):
            idx = idxs[i]

            if i == 0:  # top-left
                s_y, s_x = 0, 0
            elif i == 1:  # top-right
                s_y, s_x = 0, new_w
            elif i == 2:  # bottom-left
                s_y, s_x = new_h, 0
            else:
                s_y, s_x = new_h, new_w

            if use_crowd_mask:
                img, annot, mask = self.get_raw_data_with_crowd_mask(idx)
                new_mask = cv2.resize((mask * 255).astype(np.uint8), (new_w, new_h)) > 0.5
                mosaic_mask[s_y : s_y + new_h, s_x : s_x + new_w] = new_mask
            else:
                img, annot = self.get_raw_data(idx)

            img_h, img_w = img.shape[:2]
            new_img = cv2.resize(img, (new_w, new_h))
            mosaic_img[s_y : s_y + new_h, s_x : s_x + new_w] = new_img

            scale_y, scale_x = new_h / img_h, new_w / img_w
            objects = []
            for obj in annot:
                bbox = obj["bbox"]
                bbox[0] = int(bbox[0] * scale_x + s_x)
                bbox[2] = int(bbox[2] * scale_x + s_x)
                bbox[1] = int(bbox[1] * scale_y + s_y)
                bbox[3] = int(bbox[3] * scale_y + s_y)

                kpts, num_keypoints = None, None
                if use_keypoints:
                    kpts = np.array(obj["keypoints"]).reshape([-1, 3])
                    num_keypoints = obj["num_keypoints"]
                    vis_mask = kpts[:, 2] <= 0
                    kpts[:, 0] = kpts[:, 0] * scale_x + s_x
                    kpts[:, 1] = kpts[:, 1] * scale_y + s_y
                    kpts[vis_mask] = kpts[vis_mask] * 0

                segmentation = None
                if use_segmentation:
                    segmentation = obj["segmentation"]
                    if isinstance(segmentation, dict):
                        segmentation = mask_to_polygons(
                            coco_polygons_to_mask(segmentation, img_h, img_w)
                        )
                    for j in range(len(segmentation)):
                        seg = np.array(segmentation[j])
                        seg[::2] = seg[::2] * scale_x + s_x
                        seg[1::2] = seg[1::2] * scale_y + s_y
                        segmentation[j] = seg.astype(np.int32).tolist()

                objects.append(
                    {
                        "bbox": bbox,
                        "iscrowd": obj["iscrowd"],
                        "keypoints": kpts,
                        "num_keypoints": num_keypoints,
                        "segmentation": segmentation,
                    }
                )

            mosaic_annot.extend(objects)
        return mosaic_img, mosaic_annot, mosaic_mask

    def __len__(self) -> int:
        return len(self.annots_filepaths)


class CocoInstancesDataset(CocoDataset):
    name: str = "COCO"
    task: _tasks = "instances"
