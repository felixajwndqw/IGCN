import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch import nn
import numpy as np
import math
from .vis import FilterPlot


class GaborFunction(Function):
    r"""Extends autograd Function to create a Gabor filter with learnable theta.
    """

    @staticmethod
    def forward(ctx, input, weight):
        r"""Applies a Gabor filter to given input. Weight contains thetas.

        Args:
            input (Tensor): data to apply filter to.
            weight (Tensor): theta parameters. Must have weight.size() = [N, 2]
        """
        output = gabor(input, weight).unsqueeze(1).unsqueeze(1)
        ctx.save_for_backward(input, weight, output)
        return match_shape(output * input, input)

    @staticmethod
    def backward(ctx, grad_output):
        r"""Computes gradients for Gabor filter backprop.

        Args:
            grad_output (Tensor): gradient from graph.
        """
        input, weight, result = ctx.saved_tensors
        grad_weight = gabor_gradient(input, weight).unsqueeze_(2).unsqueeze_(2)
        grad_output = match_shape(grad_output, input, False)
        print(input.size(), grad_weight.size(), grad_output.size())
        return result*grad_output, (input*grad_weight*grad_output).permute(5, 4, 3, 2, 0, 1)


def match_shape(x, y, compress=True):
    if compress:
        x = x.view(-1, *y.size()[1:])
    else:
        x = x.view(y.size(0), -1, *x.size()[1:])
    return x


class IGConv(_ConvNd):
    def __init__(self, input_features, output_features, kernel_size,
                 stride=1, padding=0, dilation=1, bias=None, no_g=2):
        if output_features % no_g:
            raise ValueError("Number of filters (" + str(no_g) +
                             ") does not divide output features ("
                             + str(output_features) + ")")
        output_features //= no_g
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super(IGConv, self).__init__(
            input_features, output_features, kernel_size,
            stride, padding, dilation, False, (0, 0), 1, bias
        )
        self.gabor_params = nn.Parameter(torch.Tensor(2, no_g)).cuda()
        self.need_bias = (bias is not None)
        self.GaborFunction = GaborFunction.apply
        self.plot = FilterPlot(no_g)

    def forward(self, x):
        enhanced_weight = self.GaborFunction(self.weight, self.gabor_params)
        
        out = F.conv2d(x, enhanced_weight, None, self.stride,
                        self.padding, self.dilation)
        # print(self.weight.size())
        # print(enhanced_weight.size())
        # self.plot.update(self.weight, self. )
        return out
    
    


def gabor(weight, params):
    h = weight.size(2)
    w = weight.size(3)
    [x, y] = torch.Tensor(np.meshgrid(np.arange(-h/2, h/2), np.arange(-w/2, w/2)))
    if weight.is_cuda:
        x = x.cuda()
        y = y.cuda()
    return f_h(x, y) * s_h(x, y, params[0], params[1])


def gabor_gradient(weight, params):
    h = weight.size(2)
    w = weight.size(3)
    [x, y] = torch.Tensor(np.meshgrid(np.arange(-h/2, h/2), np.arange(-w/2, w/2)))
    if weight.is_cuda:
        x = x.cuda()
        y = y.cuda()
    return f_h(x, y) * d_s_h(x, y, params[0], params[1])


def f_h(x, y, sigma=math.pi):
    return torch.exp(-(x ** 2 + y ** 2) / (2*sigma**2))[np.newaxis, :]


def s_h(x, y, theta, l):
    l.unsqueeze_(1).unsqueeze_(1)
    return torch.cos(2 * math.pi / l * x_prime(x, y, theta))


def d_s_h(x, y, theta, l):
    l.unsqueeze_(1).unsqueeze_(1)
    a = -2 * math.pi / l * y_prime(x, y, theta) *\
            torch.sin(2 * math.pi / l * x_prime(x, y, theta))
    b = 2 * math.pi / l ** 2 * x_prime(x, y, theta) *\
            torch.sin(2 * math.pi / l * x_prime(x, y, theta))
    return torch.stack([a, b])


def x_prime(x, y, theta):
    return torch.cos(theta)[:, np.newaxis, np.newaxis] * x +\
           torch.sin(theta)[:, np.newaxis, np.newaxis] * y


def y_prime(x, y, theta):
    return torch.cos(theta)[:, np.newaxis, np.newaxis] * y -\
           torch.sin(theta)[:, np.newaxis, np.newaxis] * x