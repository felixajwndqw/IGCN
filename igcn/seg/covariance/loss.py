import torch


class WeightedSum(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def __call__(self, x, y):
        return x * self.c + y * (1 - self.c)


class SegRegLoss(torch.nn.Module):
    def __init__(self,
                 seg_opt=torch.nn.MSELoss(),
                 reg_opt=torch.nn.MSELoss(),
                 comb_fn=WeightedSum(.5)):
        super().__init__()
        self.seg_opt = seg_opt
        self.reg_opt = reg_opt
        self.comb_fn = comb_fn

    def forward(self, input, target):
        seg = self.seg_opt(input[0], target[0])
        reg = self.reg_opt(input[1], target[1])
        # print(input[1])
        print(input)
        # print(f'seg_loss={seg}, reg_loss={reg}')
        return self.comb_fn(seg,
                            reg)
