import glob
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm.auto import tqdm

from src.base.datasets import BaseImageDataset
from src.datasets.coco.sample import CocoAnnotation, CocoSample, create_mosaic_sample
from src.datasets.coco.transforms import ComposeCocoTransform
from src.datasets.coco.utils import COCO_KPT_LABELS, COCO_KPT_LIMBS, COCO_OBJ_LABELS
from src.logger.pylogger import log
from src.utils.files import save_yaml
from src.utils.training import get_rank

_tasks = Literal["instances", "person_keypoints"]


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
        transform: ComposeCocoTransform | None = None,
        size: int | tuple[int, int] = 512,
        mosaic_probability: float = 0,
    ):
        self.root = root
        self.split = split
        self.is_train = "train" in split
        self.transform = transform
        self.size = (size, size) if isinstance(size, int) else size
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
        crowd_mask = np.load(self.crowd_masks_filepaths[idx])  # bool
        return (crowd_mask * 255).astype(np.uint8)

    def get_raw_sample(self, idx: int) -> CocoSample:
        image = self.load_image(idx)
        annot = self.load_annot(idx)
        crowd_mask = self.load_crowd_mask(idx)
        sample = CocoSample.from_annot(image, annot, crowd_mask)
        return sample

    def get_raw_or_mosaic_sample(self, idx: int) -> CocoSample:
        sample_mosaic = random.random() < self.mosaic_probability
        get_data = self.get_raw_mosaic_sample if sample_mosaic else self.get_raw_sample
        return get_data(idx)

    def get_raw_mosaic_sample(self, idx: int) -> CocoSample:
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        samples = [self.get_raw_sample(idx) for idx in idxs]
        mosaic_sample = create_mosaic_sample(
            samples, pre_transform=self.transform.pre_mosaic, size=max(list(self.size))
        )
        return mosaic_sample

    def __getitem__(self, idx: int) -> CocoSample:
        sample = self.get_raw_or_mosaic_sample(idx)
        sample.filter_by(sample.is_crowd_obj)
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
