from abc import ABC, abstractmethod

import numpy as np


class MaskAnnot(ABC):
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def horizontal_flip(self, img_w: int, flip_idx: list[int] | None = None):
        raise NotImplementedError()

    @abstractmethod
    def vertical_flip(self, img_h: int, flip_idx: list[int] | None = None):
        raise NotImplementedError()

    def plot(self, scene: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
