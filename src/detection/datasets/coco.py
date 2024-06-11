import random

import numpy as np
from torch import Tensor

from src.datasets.coco import CocoInstancesDataset
from src.detection.transforms import ComposeDetectionTransform, DetectionTransform
from src.detection.visualization import plot_bboxes
from src.utils.image import get_color, make_grid, put_txt, stack_horizontally
from src.utils.utils import get_rank


class CocoDetectionDataset(CocoInstancesDataset):
    transform: ComposeDetectionTransform

    def plot(self, idx: int) -> np.ndarray:
        img, bbox_annots = self[idx]
        img_npy = self.inverse_preprocessing(img)
        img_npy = plot_bboxes(img_npy, bbox_annots)
        raw_image = self.plot_raw(idx, max_size=img_npy.shape[0])
        sample_vis = stack_horizontally([raw_image, img_npy])
        return sample_vis

    def __getitem__(self, idx: int) -> tuple[np.ndarray | Tensor, list]:
        if random.random() < self.mosaic_probability:
            img, annots, _ = self.get_raw_mosaiced_data(idx)
        else:
            img, annots = self.get_raw_data(idx)

        bbox_annots = [obj["bbox"] for obj in annots]
        if self.transform is not None:
            img, bbox_annots = self.transform(img, bbox_annots)
        return img, bbox_annots


if __name__ == "__main__":
    from PIL import Image

    from src.detection.transforms import DetectionTransform

    out_size = 512
    transform = DetectionTransform(out_size)
    # ds = CocoKeypointsDataset("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = CocoDetectionDataset(
        "data/COCO/raw",
        "train2017",
        transform.train,
        out_size,
        mosaic_probability=0.25,
    )
    grid = ds.plot_examples([0, 1, 2, 3, 4, 5, 6], nrows=1, stage_idxs=[1, 0])
    Image.fromarray(grid).save("test.jpg")
    # ds.explore(0)
