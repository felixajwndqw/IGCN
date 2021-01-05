import torch


class WeightedSum(torch.nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, x, y):
        return x * self.c + y * (1 - self.c)


class SegRegLoss(torch.nn.Module):
    def __init__(self,
                 seg_criterion=torch.nn.MSELoss(),
                 reg_criterion=torch.nn.MSELoss(),
                 comb_fn=WeightedSum()):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.reg_criterion = reg_criterion
        self.comb_fn = comb_fn

    def forward(self, input, target):
        seg_loss = self.seg_criterion(input[0], target[0])
        reg_loss = self.reg_criterion(input[1], target[1])
        # print(input)
        # print(f'seg_loss={seg_loss}, reg_loss={reg_loss}')
        return seg_loss + reg_loss
        # return self.comb_fn(seg_loss,
        #                     reg_loss)
