import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch import nn
import numpy as np
import math


class GaborFunction(Function):
    r"""Extends autograd Function to create a Gabor filter with learnable theta.
    """

    @staticmethod
    def forward(ctx, input, weight):
        r"""Applies a Gabor filter to given input. Weight contains thetas.

        Args:
            input (Tensor): data to apply filter to.
            weight (Tensor): theta parameters. Must have weight.size() = [N]
        """
        output = gabor(input, weight)
        ctx.save_for_backward(input, weight, output)
        return output * input

    @staticmethod
    def backward(ctx, grad_output):
        r"""Computes gradients for Gabor filter backprop.

        Args:
            grad_output (Tensor): gradient from graph.
        """
        input, weight, result = ctx.saved_tensors
        grad_weight = gabor_gradient(input, weight)
        return result*grad_output, (input*grad_weight*grad_output).transpose(0, 2)


class IGConv(_ConvNd):
    def __init__(self, input_features, output_features, kernel_size,
                 stride=1, padding=0, dilation=1, bias=None, no_g=2):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super(IGConv, self).__init__(
            input_features, output_features, kernel_size,
            stride, padding, dilation, False, (0, 0), 1, bias
        )
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.DoubleTensor(output_features, input_features))
        self.thetas = nn.Parameter(torch.DoubleTensor(no_g))
        self.need_bias = (bias is not None)
        self.GaborFunction = GaborFunction.apply

    def forward(self, x):
        enhanced_weight = self.GaborFunction(self.weight, self.thetas)
        return F.conv2d(x, enhanced_weight, None, self.stride,
                        self.padding, self.dilation)


def gabor(weight, thetas):
    h = weight.size(1)
    w = weight.size(0)
    [x, y] = torch.DoubleTensor(np.meshgrid(np.arange(-h/2, h/2), np.arange(-w/2, w/2))).cuda()
    return f_h(x, y) * s_h(x, y, thetas)


def gabor_gradient(weight, thetas):
    h = weight.size(1)
    w = weight.size(0)
    [x, y] = torch.DoubleTensor(np.meshgrid(np.arange(-h/2, h/2), np.arange(-w/2, w/2))).cuda()
    return f_h(x, y) * d_s_h(x, y, thetas)


def f_h(x, y, sigma=math.pi):
    return torch.exp(-(x ** 2 + y ** 2) / (2*sigma**2))[np.newaxis, :]


def s_h(x, y, theta):
    return torch.cos(torch.cos(theta)[:, np.newaxis, np.newaxis] * x +
                     torch.sin(theta)[:, np.newaxis, np.newaxis] * y)


def d_s_h(x, y, theta):
    return (x * torch.sin(theta)[:, np.newaxis, np.newaxis] - y * torch.cos(theta)[:, np.newaxis, np.newaxis]) *\
            torch.sin(x * torch.cos(theta)[:, np.newaxis, np.newaxis] + y * torch.sin(theta)[:, np.newaxis, np.newaxis])