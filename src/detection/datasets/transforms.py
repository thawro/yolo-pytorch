from src.datasets.coco.dataset import Mosaic
from src.datasets.coco.transforms import (
    ComposeCocoTransform,
    FormatForDetection,
    LetterBox,
    RandomAffine,
    RandomFlip,
    RandomHSV,
    Resize,
)


class DetectionTransform:
    def __init__(
        self,
        size: int,
        scale_min: float = 0.5,
        scale_max: float = 1.5,
        degrees: int = 0,
        translate: float = 0.1,
        flip_lr: float = 0.5,
        flip_ud: float = 0,
    ):
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.degrees = degrees
        self.translate = translate
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud

    def train_transform(self, dataset) -> ComposeCocoTransform:
        return ComposeCocoTransform(
            [
                Mosaic(dataset, self.size, n=4, p=1),
                # LetterBox(size=self.size),
                # Resize(self.size),
                RandomAffine(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    shear=0,
                    perspective=0,
                ),
                RandomHSV(),
                RandomFlip(p=self.flip_lr, orientation="horizontal"),
                RandomFlip(p=self.flip_ud, orientation="vertical"),
                FormatForDetection(),
            ],
        )

    def inference_transform(self) -> ComposeCocoTransform:
        return ComposeCocoTransform(
            [
                LetterBox(size=self.size),
                FormatForDetection(),
            ]
        )
