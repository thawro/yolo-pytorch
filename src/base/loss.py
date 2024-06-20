"""Implementation of loss Base and weighted loss classes"""

from abc import abstractmethod

from torch import Tensor
from torch.nn.modules.loss import _Loss


class BaseLoss(_Loss):
    @abstractmethod
    def calculate_loss(self, **kwargs) -> Tensor:
        raise NotImplementedError()
