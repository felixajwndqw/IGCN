import torch
import torch.nn as nn


class RCFLoss(nn.Module):
    """Loss for RCF style model.

    Loss is weighted on per-pixel basis according to class balance and params.

    Parameters:
        eta (float, optional): Consensus proporition for positive sample.
        lambda (float, optional): Controls positive class weighting (not negative).
    """
    def __init__(self, eta=.5, lambd=1.1):
        super().__init__()
        self.eta = eta
        self.lambd = lambd
        self.seg_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        target[target >= self.eta] = 1.
        weight = self.loss_weight(target)

        if not (type(output) == tuple or type(output) == list):
            return (self.seg_loss(output, target) * weight).mean()
        seg_losses = [
            (self.seg_loss(seg, target) * weight).mean()
            for seg in output
        ]
        return sum(seg_losses)

    def loss_weight(self, target):
        p = torch.sum(target >= self.eta)
        n = torch.sum(target == 0)
        weight = torch.zeros_like(target)
        weight[target >= self.eta] = n / (p + n)
        weight[target == 0] = self.lambd * p / (p + n)
        return weight
