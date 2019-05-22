import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch import nn
import numpy as np
import math
from .vis import FilterPlot, FilterPlotNew
from .rot_pool import RotMaxPool2d


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
        grad_output = match_shape(grad_output, grad_weight, False)
        # print(input.size(), grad_weight.size(), grad_output.size())
        return result*grad_output, (input*grad_weight*grad_output).permute(5, 4, 3, 2, 0, 1)


def match_shape(x, y, compress=True):
    if compress:
        x = x.view(-1, *y.size()[1:])
    else:
        x = x.view(y.size(1), -1, *x.size()[1:])
    return x


class IGConv(_ConvNd):
    def __init__(self, input_features, output_features, kernel_size,
                 stride=1, padding=0, dilation=1, bias=None,
                 rot_pool=False, no_g=2, pool_stride=1, plot=False,
                 max_gabor=True):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super(IGConv, self).__init__(
            input_features, output_features, kernel_size,
            stride, padding, dilation, False, (0, 0), 1, bias, 'zeros'
        )
        self.gabor_params = nn.Parameter(data=torch.Tensor(2, no_g))
        self.gabor_params.data[0] = torch.arange(no_g, dtype=torch.float) / (no_g - 1) * math.pi
        self.gabor_params.data[1].uniform_(-1 / math.sqrt(no_g), 1 / math.sqrt(no_g))
        self.need_bias = bias is not None
        self.register_parameter(name="gabor", param=self.gabor_params)
        self.GaborFunction = GaborFunction.apply
        self.no_g = no_g
        self.rot_pool = rot_pool
        self.max_gabor = max_gabor
        self.pooling = []
        if rot_pool:
            self.pooling = RotMaxPool2d(kernel_size=3, stride=pool_stride)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=pool_stride)
        self.bn = nn.BatchNorm2d(output_features * no_g)

        if plot:
            self.plot = FilterPlotNew(no_g, kernel_size[0])
        else:
            self.plot = None

    def forward(self, x):
        enhanced_weight = self.GaborFunction(self.weight, self.gabor_params)
        out = F.conv2d(x, enhanced_weight, None, self.stride,
                       self.padding, self.dilation)
        out = self.bn(out)
        if self.rot_pool:
            pool_out = self.pooling(out, self.gabor_params[0, :])
        else:
            pool_out = self.pooling(out)

        if self.max_gabor:
            pool_out = pool_out.view(pool_out.size(0), enhanced_weight.size(0) // self.no_g, self.no_g, pool_out.size(2), pool_out.size(3))
            pool_out, _ = torch.max(pool_out, dim=2)

        if self.plot is not None:
            # print(["{:.2f}, {:.2f}".format(t.item(), l.item()) for t, l in self.gabor_params.data.transpose(1, 0)])
            self.plot.update(self.weight[0, 0].data.cpu().numpy(),
                             enhanced_weight[::(enhanced_weight.size(0) // self.no_g), 0].detach().cpu().numpy(),
                             self.gabor_params.data.cpu().numpy())

        return pool_out


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
    dt = -2 * math.pi / l * y_prime(x, y, theta) *\
            torch.sin(2 * math.pi / l * x_prime(x, y, theta))
    dl = 2 * math.pi / l ** 2 * x_prime(x, y, theta) *\
            torch.sin(2 * math.pi / l * x_prime(x, y, theta))
    return torch.stack([dt, dl])


def x_prime(x, y, theta):
    return torch.cos(theta)[:, np.newaxis, np.newaxis] * x +\
           torch.sin(theta)[:, np.newaxis, np.newaxis] * y


def y_prime(x, y, theta):
    return torch.cos(theta)[:, np.newaxis, np.newaxis] * y -\
           torch.sin(theta)[:, np.newaxis, np.newaxis] * x