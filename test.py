import igcn
import torch
import numpy as np
from matplotlib import pyplot as plt
import math


def show_ims(ims):
    ims = ims.detach()
    fig = plt.figure(figsize=(8, 8))
    grid = math.ceil(math.sqrt(ims.size(0)))
    for i in range(0, ims.size(0)):
        fig.add_subplot(grid, grid, i+1)
        plt.imshow(ims[i], cmap="gray")
    plt.show()


def gabor_filter_demo(no_g=9):
    weight = torch.ones(10, 10, dtype=torch.double, requires_grad=True)
    thetas = torch.DoubleTensor(torch.arange(no_g, dtype=torch.double)/no_g * 3.14)
    thetas = torch.autograd.Variable(thetas, requires_grad=True)
    gabor = igcn.GaborFunction.apply
    out = gabor(weight, thetas)

    show_ims(out)

