import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


class Loss_2(_Loss):
    def forward(self, preds: Tensor, targets: Tensor):
        pass


class Loss_1(_Loss):
    def forward(self, preds: Tensor, targets: Tensor):
        pass


class DetectionLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()
        self.loss_1 = Loss_1()
        self.loss_2 = Loss_2()

    def calculate_loss(
        self,
        preds: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        loss_1 = self.loss_1(preds, targets)
        loss_2 = self.loss_2(preds, targets)
        return loss_1, loss_2
