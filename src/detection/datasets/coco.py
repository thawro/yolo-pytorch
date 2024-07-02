import numpy as np
import torch
from torch import Tensor

from src.annots.ops import xywh2xyxy
from src.annots.visuzalization import plot_boxes
from src.datasets.coco import CocoInstancesDataset, CocoSample
from src.detection.datasets.transforms import ComposeCocoTransform, DetectionTransform
from src.utils.image import stack_horizontally


class CocoDetectionDataset(CocoInstancesDataset):
    transform: ComposeCocoTransform

    @staticmethod
    def collate_fn(batch: list[CocoSample]) -> dict[str, Tensor]:
        """Collates data samples into batches."""
        batch_dct = [sample.to_batch() for sample in batch]
        del batch
        keys = batch_dct[0].keys()
        values = list(zip(*[list(b.values()) for b in batch_dct]))
        new_batch = {}
        for i, k in enumerate(keys):
            value = values[i]
            if k == "images":
                value = torch.from_numpy(np.stack(value, 0))
            elif k in {"masks", "kpts", "boxes_xywh", "classes", "segments", "obb"}:
                value = torch.from_numpy(np.concatenate(value, 0))
            elif k == "num_obj":
                value = torch.cat([torch.zeros(num_obj) + i for i, num_obj in enumerate(value)])
                k = "batch_idxs"
            elif k in {"scale_wh", "pad_tlbr"}:
                value = torch.from_numpy(np.array(value))
            new_batch[k] = value
        return new_batch

    def plot(self, idx: int, **kwargs) -> np.ndarray:
        sample = self[idx]
        image, boxes_xywh, classes = sample.image, sample.boxes.data, sample.classes
        h, w = image.shape[:2]
        boxes_xywh[:, 0::2] *= w  # unnormalize
        boxes_xywh[:, 1::2] *= h  # unnormalize
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_labels = [self.objs_labels[label_id] for label_id in classes]
        image = plot_boxes(image, boxes_xyxy, boxes_labels, classes)
        raw_sample = self.get_raw_sample(idx)
        raw_image = raw_sample.plot(max_size=h, segmentation=True, keypoints=False)
        sample_vis = stack_horizontally([raw_image, image])
        return sample_vis


if __name__ == "__main__":
    from PIL import Image

    from src.detection.datasets.transforms import DetectionTransform

    size = 640
    transform = DetectionTransform(size)
    # ds = CocoKeypointsDataset("data/COCO", "val2017", transform.inference, out_size, hm_resolutions)
    ds = CocoDetectionDataset(
        "data/COCO",
        "train2017",
        size,
    )
    ds.set_transform(transform.train_transform(ds))
    ds.explore()
    grid = ds.plot_examples([0, 55, 2, 3, 4, 5, 6])
    Image.fromarray(grid).save("test.jpg")


# TODO: add merge samples (mosaic, cutoff, copypaste, mixup)
