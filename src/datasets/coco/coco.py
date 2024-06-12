import glob
import os
import random
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.datasets.coco.sample import CocoAnnotation, CocoSample
from src.datasets.coco.utils import (
    COCO_KPT_LABELS,
    COCO_KPT_LIMBS,
    COCO_OBJ_LABELS,
    coco_polygons_to_mask,
    mask_to_polygons,
)
from src.logger.pylogger import log
from src.utils.files import load_yaml, save_yaml
from src.utils.utils import get_rank

_tasks = Literal["instances", "person_keypoints"]


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
        size: int = 512,
        mosaic_probability: float = 0,
    ):
        self.root = root
        self.split = split
        self.is_train = "train" in split
        self.transform = transform
        self.size = size
        self.mosaic_probability = mosaic_probability
        self.task_split = f"{self.task}_{self.split}"

        self.images_dir = f"{self.root}/images/{self.split}"
        self.annots_dir = f"{self.root}/annotations/{self.task_split}"
        self.crowd_masks_dir = f"{self.root}/crowd_masks/{self.split}"
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
            path.replace(".yaml", ".npy")
            .replace("annotations/", "crowd_masks/")
            .replace(f"{self.task}_", "")
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
            crowd_mask_filepath = f"{self.crowd_masks_dir}/{filename}.npy"
            meta = {
                "height": img_info["height"],
                "width": img_info["width"],
                "annot_filepath": annot_filepath,
                "image_filepath": img_filepath,
                "crowd_mask_filepath": crowd_mask_filepath,
            }
            annot_to_save = {"objects": annot, "meta": meta}
            save_yaml(annot_to_save, annot_filepath)
            annot = CocoAnnotation.from_coco_annot(annot)
            crowd_mask = annot.get_crowd_mask(img_info["height"], img_info["width"])
            np.save(crowd_mask_filepath, crowd_mask)

    def load_annot(self, idx: int) -> CocoAnnotation:
        annot = CocoAnnotation.from_yaml(self.annots_filepaths[idx])
        return annot

    def load_crowd_mask(self, idx: int) -> np.ndarray:
        return np.load(self.crowd_masks_filepaths[idx])

    def get_raw_sample(self, idx: int) -> CocoSample:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        crowd_mask = self.load_crowd_mask(idx)
        sample = CocoSample(image, annot, crowd_mask)
        return sample

    def get_raw_or_mosaic_sample(self, idx: int) -> CocoSample:
        sample_mosaic = random.random() < self.mosaic_probability
        get_data = self.get_raw_mosaic_sample if sample_mosaic else self.get_raw_sample
        return get_data(idx)

    def get_raw_mosaic_sample(self, idx: int) -> CocoSample:
        out_size = self.size * 2
        img_size = out_size // 2
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        mosaic_annot = []
        mosaic_img = np.zeros([out_size, out_size, 3], dtype=np.uint8)
        mosaic_crowd_mask = np.empty([out_size, out_size], dtype=np.bool_)

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

            img, crowd_mask, annot = self.get_raw_data(idx)
            new_crowd_mask = cv2.resize((crowd_mask * 255).astype(np.uint8), (new_w, new_h)) > 0.5
            mosaic_crowd_mask[s_y : s_y + new_h, s_x : s_x + new_w] = new_crowd_mask

            img_h, img_w = img.shape[:2]
            new_img = cv2.resize(img, (new_w, new_h))
            mosaic_img[s_y : s_y + new_h, s_x : s_x + new_w] = new_img

            scale_y, scale_x = new_h / img_h, new_w / img_w
            objects = []
            for obj in annot.objects:
                box_xyxy = obj.box_xyxy
                box_xyxy[0] = box_xyxy[0] * scale_x + s_x
                box_xyxy[1] = box_xyxy[1] * scale_y + s_y
                box_xyxy[2] = box_xyxy[2] * scale_x + s_x
                box_xyxy[3] = box_xyxy[3] * scale_y + s_y

                kpts = None
                if obj.has_keypoints():
                    kpts = np.array(obj.keypoints).reshape([-1, 3])
                    vis_mask = kpts[:, 2] <= 0
                    kpts[:, 0] = kpts[:, 0] * scale_x + s_x
                    kpts[:, 1] = kpts[:, 1] * scale_y + s_y
                    kpts[vis_mask] = kpts[vis_mask] * 0

                segmentation = None
                if obj.has_segmentation():
                    segmentation = obj.segmentation
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
                    CocoObjectAnnotation(
                        area=obj.area / 4,
                        box_xyxy=box_xyxy,
                        category_id=obj.category_id,
                        id=obj.id,
                        image_id=obj.image_id,
                        iscrowd=obj.iscrowd,
                        segmentation=segmentation,
                        num_keypoints=obj.num_keypoints,
                        keypoints=kpts,
                    )
                )
            mosaic_annot.extend(objects)
        mosaic_annot = CocoAnnotation(mosaic_annot)
        return mosaic_img, mosaic_crowd_mask, mosaic_annot

    def __len__(self) -> int:
        return len(self.annots_filepaths)


class CocoInstancesDataset(CocoDataset):
    name: str = "COCO"
    task: _tasks = "instances"
