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
        images = np.stack([sample.image for sample in batch])
        samples_num_objs = [len(sample.classes) for sample in batch]
        batch_idxs = [torch.zeros(num_obj) + i for i, num_obj in enumerate(samples_num_objs)]
        boxes_xywh = np.concatenate([sample.boxes.data for sample in batch])
        classes = np.concatenate([sample.classes for sample in batch])
        # segments = np.concatenate([sample.segments for sample in batch])
        new_batch = {
            "batch_idxs": torch.cat(batch_idxs),
            "images": torch.from_numpy(images),
            "boxes_xywh": torch.from_numpy(boxes_xywh),
            "classes": torch.from_numpy(classes),
            # "segments": torch.from_numpy(segments),
        }
        return new_batch

    def plot(self, idx: int, **kwargs) -> np.ndarray:
        sample = self[idx]
        image, boxes_xywh, classes = sample.image, sample.boxes.data, sample.classes
        image_npy = self.inverse_preprocessing(image)
        h, w = image_npy.shape[:2]
        boxes_xywh[:, 0::2] *= w  # unnormalize
        boxes_xywh[:, 1::2] *= h  # unnormalize
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        boxes_labels = [self.objs_labels[label_id] for label_id in classes]
        image_npy = plot_boxes(image_npy, boxes_xyxy, boxes_labels, classes)
        raw_sample = self.get_raw_sample(idx)
        raw_image = raw_sample.plot(max_size=image_npy.shape[0], segmentation=True, keypoints=False)
        sample_vis = stack_horizontally([raw_image, image_npy])
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
        transform.train,
        size,
        mosaic_probability=1,
    )
    ds.explore()
    grid = ds.plot_examples([0, 55, 2, 3, 4, 5, 6])
    Image.fromarray(grid).save("test.jpg")


# TODO: add merge samples (mosaic, cutoff, copypaste, mixup)
