
import math
import logging
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from .igabor import gabor_cmplx
from .vis import FilterPlot
from .rot_pool import RotMaxPool2d
from .utils import _pair
from .cmplx import cmplx, conv_cmplx, relu_cmplx, bnorm_cmplx, pool_cmplx


log = logging.getLogger(__name__)


class IGaborCmplx(nn.Module):
    """Wraps the complex Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
    """
    def __init__(self, no_g=4, layer=False, kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.gabor_params = nn.Parameter(data=torch.Tensor(2, no_g))
        self.gabor_params.data[0] = torch.arange(no_g) / (no_g) * math.pi
        self.gabor_params.data[1].uniform_(-1 / math.sqrt(no_g),
                                           1 / math.sqrt(no_g))
        self.register_parameter(name="gabor", param=self.gabor_params)
        self.no_g = no_g
        self.register_buffer("gabor_filters", torch.Tensor(self.no_g,
                                                           *kernel_size))
        self.layer = layer
        self.calc_filters = True  # Flag whether filter bank needs recalculating
        self.register_backward_hook(self.set_filter_calc)

    def forward(self, x):
        log.debug(f'x.size()={x.unsqueeze(1).size()}, gabor={gabor_cmplx(x, self.gabor_params).unsqueeze(2).size()}')
        if self.calc_filters:
            self.generate_gabor_filters(x)
        out = self.gabor_filters * x.unsqueeze(1)
        out = out.view(2, -1 , *out.size()[3:])
        log.debug(f'out.size()={out.size()}')
        if self.layer:
            out = out.view(x.size(0), x.size(1) * self.no_g, *x.size()[2:])
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        self.gabor_filters = gabor_cmplx(x, self.gabor_params).unsqueeze(2)
        self.calc_filters = False

    def set_filter_calc(self, *args):
        """Called in backward hook so that filter bank will be regenerated.
        """
        self.calc_filters = True


class IGConvCmplx(nn.Module):
    """Implements a convolutional layer where weights are first Gabor modulated.

    In addition, rotated pooling, gabor pooling and batch norm are implemented
    below.
    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        kernel_size (int, tuple): Size of kernel.
        rot_pool (bool, optional):
        no_g (int, optional): The number of desired Gabor filters.
        pool_stride (int, optional):
        plot (bool, optional): Plots feature maps/weights
        max_gabor (bool, optional):
        include_gparams (bool, optional): Includes gabor params with highest
            activations as extra feature channels.
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 rot_pool=None, no_g=2, pool_stride=1, plot=False,
                 max_gabor=False, include_gparams=False, **conv_kwargs):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        kernel_size = _pair(kernel_size)
        super().__init__()
        self.ReConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        self.ImConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        self.conv = conv_cmplx

        self.gabor = IGaborCmplx(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.rot_pool = rot_pool
        self.max_gabor = max_gabor
        self.include_gparams = include_gparams
        self.conv_kwargs = conv_kwargs
        self.pooling = []
        if rot_pool:
            self.pooling = RotMaxPool2d(kernel_size=3, stride=pool_stride)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=pool_stride)
        self.bn = nn.BatchNorm2d(output_features * no_g)

        if plot:
            self.plot = FilterPlot(no_g, kernel_size[0], output_features)
        else:
            self.plot = None
        log.debug(f'stride={self.ReConv.stride}, padding={self.ReConv.padding}, dilation={self.ReConv.dilation}')

    def forward(self, x):
        enhanced_weight = self.gabor(cmplx(self.ReConv.weight, self.ImConv.weight))
        out = self.conv(x, enhanced_weight, **self.conv_kwargs)
        log.info(f'x.size()={x.size()}, '
                 f'self.ReConv.weight.size()={self.ReConv.weight.size()}, '
                 f'enhanced_weight.size()={enhanced_weight.size()}, '
                 f'out.size()={out.size()}')

        if self.plot is not None:
            self.plot.update(self.weight[:, 0].clone().detach().cpu().numpy(),
                             enhanced_weight[:, 0].clone().detach().cpu().numpy(),
                             self.gabor_params.clone().detach().cpu().numpy())

        max_out = None
        if self.max_gabor or self.include_gparams:
            max_out = out.view(2,
                               out.size(1),
                               enhanced_weight.size(1) // self.no_g,
                               self.no_g,
                               out.size(3),
                               out.size(4))
            max_out, max_idxs = torch.max(max_out, dim=3)
            # max_gparams = max_gparams.permute(1, 2, 3, 0, 4, 5)
            # print(f'max_gparams.size()={max_gparams.size()}')

        if self.max_gabor:
            out = max_out

        if self.include_gparams:
            max_gparams = self.gabor.gabor_params[0, max_idxs]
            out = torch.stack(out, max_gparams, dim=3)

        return out


class ReLUCmplx(nn.Module):
    """Implements complex rectified linear unit.
    """
    def __init__(self, eps=1e-8, inplace=False):
        super().__init__()
        self.eps = eps
        self.inplace = inplace

    def forward(self, x):
        return relu_cmplx(x, inplace=self.inplace)


class BatchNormCmplx(nn.Module):
    """Implements complex batch normalisation.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return bnorm_cmplx(x, self.eps)


class MaxPoolCmplx(nn.Module):
    """Implements complex max pooling.
    """
    def __init__(self, kernel_size, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, **self.pool_kwargs)
