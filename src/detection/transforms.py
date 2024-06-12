from src.base.transforms import ImageTransform
from src.datasets.coco.transforms import (
    ComposeCocoTransform,
    LetterBox,
    Normalize,
    RandomAffine,
    RandomHorizontalFlip,
    ToTensor,
)


class DetectionTransform(ImageTransform):
    def __init__(
        self,
        size: int | tuple[int, int],
        scale: float = 0.75,
        degrees: int = 0,
        translate: float = 0.1,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__(size, mean, std)
        self.train = ComposeCocoTransform(
            [
                RandomHorizontalFlip(p=0.5),
                RandomAffine(
                    degrees=degrees, translate=translate, scale=scale, shear=0, perspective=0
                ),
                LetterBox(size=size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
        self.inference = ComposeCocoTransform(
            [
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
