import igcn
import torch
import math
from quicktorch.utils import imshow


def gabor_filter_demo(no_g=9):
    weight = torch.ones(1, 1, 10, 10, dtype=torch.double, requires_grad=True)
    thetas = torch.DoubleTensor(2, no_g)
    thetas[0] = torch.DoubleTensor(torch.arange(no_g, dtype=torch.double)/no_g * 3.14)
    thetas[1].uniform_(-1 / math.sqrt(no_g), 1 / math.sqrt(no_g))
    thetas = torch.autograd.Variable(thetas, requires_grad=True)
    gabor = igcn.GaborFunction.apply
    out = gabor(weight, thetas)
    imshow(out.detach())


if __name__ == '__main__':
    gabor_filter_demo()
