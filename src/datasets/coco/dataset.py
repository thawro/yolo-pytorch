import gc
import glob
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from pycocotools.coco import COCO
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.datasets.coco.sample import CocoAnnotation, CocoSample
from src.datasets.coco.transforms import BORDER_VALUE, LetterBox, Resize
from src.datasets.coco.utils import (
    COCO_KPT_LABELS,
    COCO_KPT_LIMBS,
    COCO_OBJ_LABELS,
    coco91_to_coco80_labels,
    coco_rle_to_seg,
)
from src.logger.pylogger import log_msg
from src.utils.files import load_yaml, save_yaml
from src.utils.training import get_rank

_tasks = Literal["instances", "person_keypoints"]


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


def load_dataset_cache_file(path):
    """Load an *.cache dictionary from path."""
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True)  # load dict
    gc.enable()
    return cache


def coco_annot_to_cached_annot(filepath: str) -> dict[str, np.ndarray | None]:
    annot = load_yaml(filepath)
    objects = annot["objects"]
    meta = annot["meta"]
    image_filepath = meta["image_filepath"]

    classes = np.array([coco91_to_coco80_labels[obj["category_id"] - 1] for obj in objects])
    iscrowd = np.array([obj["iscrowd"] for obj in objects])

    boxes_xyxy, kpts, segments = None, None, None

    if "bbox" in objects[0]:
        boxes_xyxy = []
        for obj in objects:
            x, y, w, h = obj["bbox"]
            box_xyxy = [x, y, x + w, y + h]
            boxes_xyxy.append(box_xyxy)
        boxes_xyxy = np.array(boxes_xyxy)

    if "segmentation" in objects[0]:
        segments = []
        for obj in objects:
            segmentation = obj["segmentation"]
            if isinstance(segmentation, dict):  # RLE annotation
                segmentation = coco_rle_to_seg(segmentation)
            segmentation = [np.array(seg).reshape(-1, 2) for seg in segmentation]
            segmentation = reduce_segments(segmentation)
            segmentation = resample_segmentation(segmentation, n=100)
            segments.append(segmentation)
        segments = np.array(segments)

    return {
        "boxes_xyxy": boxes_xyxy,
        "classes": classes,
        "segments": segments,
        "kpts": kpts,
        "is_crowd_obj": iscrowd,
        "image_filepath": image_filepath,
    }


class CocoDataset(BaseImageDataset):
    name: str = "COCO"
    task: _tasks
    size: tuple[int, int]
    kpts_limbs = COCO_KPT_LIMBS
    kpts_labels = COCO_KPT_LABELS
    objs_labels = COCO_OBJ_LABELS

    def __init__(
        self,
        root: str,
        split: str,
        size: int | tuple[int, int] = 512,
    ):
        self.root = root
        self.split = split
        self.is_train = "train" in split
        self.size = (size, size) if isinstance(size, int) else size
        self.task_split = f"{self.task}_{self.split}"
        self.images_dir = f"{self.root}/images/{self.split}"
        self.annots_dir = f"{self.root}/annotations/{self.task_split}"
        self.gt_annot_filepath = f"{self.root}/annotations/{self.task_split}.json"
        self.labels_cache_path = f"{self.annots_dir}.cache"
        self.crowd_masks_dir = f"{self.root}/crowd_masks/{self.split}"
        self._set_paths()
        self.images_cache = [None] * len(self.images_filepaths)
        self.cache_annots()
        self.buffer_cache = []
        self.max_buffer_length = 640
        self.transform = None

    def cache_annots(self):
        path = Path(self.labels_cache_path)
        if path.exists():
            log_msg(f"-> Loading cached labels from {str(path)} (rank={get_rank()})")
            self.cached_annots = load_dataset_cache_file(str(path))
            return
        log_msg(f"-> Creating new labels cache in {str(path)} (rank={get_rank()})")
        parallel = Parallel(n_jobs=-1, return_as="generator")
        output_generator = parallel(
            delayed(coco_annot_to_cached_annot)(path)
            for path in tqdm(self.annots_filepaths, desc=f"Caching annots (rank={get_rank()})")
        )
        self.cached_annots = list(output_generator)
        np.save(str(path), self.cached_annots)
        path.with_suffix(".cache.npy").rename(path)

    def set_transform(self, transform):
        self.transform = transform

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
            log_msg(
                f"\tCurrent process (rank = {rank}) is not the main process (rank = 0) -> Skipping annots files saving",
                logging.WARN,
            )
            return
        annot_dir_exists = os.path.exists(self.annots_dir)
        num_files = len(glob.glob(f"{self.annots_dir}/*"))

        coco = COCO(f"{self.root}/annotations/{self.task_split}.json")
        ids = list(coco.imgs.keys())
        # remove_images_without_annotations
        n_before = len(ids)
        log_msg("-> Removing data samples without annotations")
        ids = [_id for _id in ids if len(coco.getAnnIds(imgIds=_id, iscrowd=None)) > 0]
        log_msg(f"before = {n_before}, after = {len(ids)}")
        num_gt_annots = len(ids)
        if annot_dir_exists and num_files == num_gt_annots:
            log_msg(
                f"-> {self.task_split} annotations already saved to files -> Skipping annots files saving"
            )
            return
        if num_files != num_gt_annots and annot_dir_exists:
            log_msg(
                f"Number of ground truth annotations ({num_gt_annots}) is not equal to "
                f"number of saves annotation files ({num_files}). "
                f"Repeating the annotations files saving process"
            )
        log_msg(
            f"-> Saving {self.task_split} annotations to files (annots to .yaml and crowd masks to .npy)"
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
            # TODO: make it possible without CocoAnnotation
            annot = CocoAnnotation.from_coco_annot(annot)
            crowd_mask = annot.get_crowd_mask(img_info["height"], img_info["width"])
            np.save(crowd_mask_filepath, crowd_mask)

    def load_annot(self, idx: int) -> dict:
        return deepcopy(self.cached_annots[idx])

    def load_crowd_mask(self, idx: int) -> np.ndarray:
        crowd_mask = np.load(self.crowd_masks_filepaths[idx])  # bool
        return (crowd_mask * 255).astype(np.uint8)

    def load_image(self, idx: int) -> np.ndarray:
        # return super().load_image(idx)
        image = self.images_cache[idx]
        if image is None:
            image = super().load_image(idx)
            self.images_cache[idx] = image
            self.buffer_cache.append(idx)
            if 1 < len(self.buffer_cache) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer_cache.pop(0)
                self.images_cache[j] = None
        return image

    def get_raw_sample(self, idx: int) -> CocoSample:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        crowd_mask = self.load_crowd_mask(idx)
        sample = CocoSample(image=image, crowd_mask=crowd_mask, **annot)
        return sample

    def __getitem__(self, idx: int) -> CocoSample:
        sample = self.get_raw_sample(idx)
        # sample.filter_by(sample.is_crowd_obj)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def collate_fn(batch: list[CocoSample]) -> dict[str, torch.Tensor]:
        """Collates data samples into batches."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.annots_filepaths)


class CocoInstancesDataset(CocoDataset):
    name: str = "COCO"
    task: _tasks = "instances"


def create_mosaic_sample(samples: list[CocoSample], size: int = 640) -> CocoSample:
    assert len(samples) == 4
    mosaic_size = size * 2
    mosaic_boxes = []
    mosaic_classes = []
    mosaic_kpts = []
    mosaic_segments = []
    mosaic_is_crowd_obj = []
    mosaic_image = np.full([mosaic_size, mosaic_size, 3], BORDER_VALUE[0], dtype=np.uint8)
    mosaic_crowd_mask = np.empty([mosaic_size, mosaic_size], dtype=np.uint8)
    xc, yc = size, size
    for i, sample in enumerate(samples):
        h, w = sample.image.shape[:2]
        if i == 0:  # top-left
            # xmin, ymin = 0, 0
            xmin, ymin = xc - w, yc - h
        elif i == 1:  # top-right
            # xmin, ymin = xc, 0
            xmin, ymin = xc, yc - h
        elif i == 2:  # bottom-left
            # xmin, ymin = 0, yc
            xmin, ymin = xc - w, yc
        else:  # bottom-right
            # xmin, ymin = xc, yc
            xmin, ymin = xc, yc
        ymax = ymin + h
        xmax = xmin + w

        mosaic_image[ymin:ymax, xmin:xmax] = sample.image
        if sample.crowd_mask is not None:
            mosaic_crowd_mask[ymin:ymax, xmin:xmax] = sample.crowd_mask

        sample.add_coords(xmin, ymin)
        sample.convert_boxes("xyxy")
        mosaic_boxes.append(sample.boxes.data)
        mosaic_segments.append(sample.segments.data)
        mosaic_kpts.append(sample.kpts.data)
        mosaic_classes.append(sample.classes)
        mosaic_is_crowd_obj.append(sample.is_crowd_obj)

    def cat_or_none(arrays: list[np.ndarray]) -> np.ndarray | None:
        if any(el is None for el in arrays):
            return None
        return np.concatenate(arrays)

    return CocoSample(
        image=mosaic_image,
        boxes_xyxy=cat_or_none(mosaic_boxes),
        classes=cat_or_none(mosaic_classes),
        kpts=cat_or_none(mosaic_kpts),
        segments=cat_or_none(mosaic_segments),
        crowd_mask=mosaic_crowd_mask,
        is_crowd_obj=cat_or_none(mosaic_is_crowd_obj),
        image_filepath="mosaic",
    )


class Mosaic:
    def __init__(self, dataset: CocoDataset, size: int = 640, n: Literal[4, 9] = 4, p: float = 1.0):
        self.dataset = dataset
        # self.pre_transform = LetterBox(size)
        self.size = size
        # self.pre_transform = Resize(max_size=size)
        self.pre_transform = None
        self.n = n
        self.p = p

    def get_indexes(self, buffer=True):
        """Return a list of random indexes from the dataset."""
        if buffer and len(self.dataset.buffer_cache) >= self.n:  # select images from buffer
            return random.choices(self.dataset.buffer_cache, k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def __call__(self, sample: CocoSample) -> CocoSample:
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return sample
        # Get index of one or three other images
        idxs = self.get_indexes()
        # Get images information will be used for Mosaic or MixUp
        mix_samples = [sample] + [self.dataset.get_raw_sample(idx) for idx in idxs]

        if self.pre_transform is not None:
            for i, sample in enumerate(mix_samples):
                mix_samples[i] = self.pre_transform(sample)

        # Mosaic or MixUp
        return create_mosaic_sample(mix_samples, size=self.size)
