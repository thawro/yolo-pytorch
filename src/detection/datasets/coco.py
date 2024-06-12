import numpy as np
from torch import Tensor

from src.datasets.coco import CocoInstancesDataset
from src.datasets.coco.sample import CocoSample
from src.detection.transforms import ComposeCocoTransform, DetectionTransform
from src.detection.visualization import plot_boxes
from src.utils.image import stack_horizontally


class CocoDetectionDataset(CocoInstancesDataset):
    transform: ComposeCocoTransform

    def plot(self, idx: int) -> np.ndarray:
        img, boxes_xyxy, boxes_labels_ids = self[idx]
        img_npy = self.inverse_preprocessing(img)
        boxes_labels = [self.objs_labels[label_id] for label_id in boxes_labels_ids]
        img_npy = plot_boxes(img_npy, boxes_xyxy, boxes_labels, boxes_labels_ids, boxes_conf=None)
        raw_sample = self.get_raw_sample(idx)
        raw_image = raw_sample.plot(max_size=img_npy.shape[0], segmentation=True, keypoints=False)
        sample_vis = stack_horizontally([raw_image, img_npy])
        return sample_vis

    def __getitem__(self, idx: int) -> tuple[np.ndarray | Tensor, np.ndarray, np.ndarray]:
        sample = self.get_raw_or_mosaic_sample(idx)
        sample.filter_by(sample.is_crowd_obj)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample.image, sample.boxes_xyxy, sample.boxes_cls


class TestCocoDetectionDataset(CocoInstancesDataset):
    transform: ComposeCocoTransform

    def plot(self, idx: int) -> np.ndarray:
        sample = self[idx]
        sample.inverse_preprocessing()
        transformed_sample_plot = sample.plot(max_size=None, boxes=True, title="Trans sample")
        max_size = sample.image.shape[0]

        sample.inverse_transform()

        raw_sample = self.get_raw_sample(idx)
        raw_sample_plot = raw_sample.plot(
            max_size=max_size, segmentation=True, boxes=True, title="Raw sample"
        )

        inv_transformed_sample_plot = sample.plot(
            max_size=max_size, boxes=True, title="Inv Trans sample"
        )

        sample_vis = stack_horizontally(
            [raw_sample_plot, transformed_sample_plot, inv_transformed_sample_plot]
        )
        return sample_vis

    def __getitem__(self, idx: int) -> CocoSample:
        sample = self.get_raw_or_mosaic_sample(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    from PIL import Image

    from src.detection.transforms import DetectionTransform

    size = 640
    transform = DetectionTransform(size)
    # ds = CocoKeypointsDataset("data/COCO/raw", "val2017", transform.inference, out_size, hm_resolutions)
    ds = TestCocoDetectionDataset(
        "data/COCO",
        "train2017",
        transform.train,
        size,
        mosaic_probability=0.0,
    )
    grid = ds.plot_examples([0, 55, 2, 3, 4, 5, 6])
    Image.fromarray(grid).save("test.jpg")
    # ds.explore(0)


# TODO: add merge samples (mosaic, cutoff, copypaste, mixup)
# TODO: add HSV aug
