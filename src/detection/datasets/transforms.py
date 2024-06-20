from src.base.transforms import ImageTransform
from src.datasets.coco.transforms import (
    ComposeCocoTransform,
    FormatForDetection,
    LetterBox,
    Normalize,
    RandomAffine,
    RandomFlip,
    RandomHSV,
    ToTensor,
)


class DetectionTransform(ImageTransform):
    def __init__(
        self,
        size: int | tuple[int, int],
        scale_min: float = 0.75,
        scale_max: float = 1.75,
        degrees: int = 0,
        translate: float = 0.1,
        flip_lr: float = 0.5,
        flip_ud: float = 0,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__(size, mean, std)
        self.size = size
        postprocessing = [
            FormatForDetection(),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
        self.train = ComposeCocoTransform(
            transforms=[
                LetterBox(size=size),
                RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    shear=0,
                    perspective=0,
                ),
                RandomHSV(),
                RandomFlip(p=flip_lr, orientation="horizontal"),
                RandomFlip(p=flip_ud, orientation="vertical"),
                *postprocessing,
            ],
            pre_mosaic=LetterBox(size=size),
        )
        self.inference = ComposeCocoTransform([LetterBox(size=size), *postprocessing])
