import torch


def to_mask(x):
    # x = x / x.max(dim=1, keepdim=True)[0]
    # return (x == 1).float()
    print(x.min(), x.max())
    x = torch.sigmoid(x).round()
    print(x.min(), x.max())
    return x
