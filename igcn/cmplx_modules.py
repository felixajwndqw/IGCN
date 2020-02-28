
import math
import logging
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from .igabor import gabor_cmplx
from .utils import _pair
from .cmplx import (
    cmplx,
    conv_cmplx,
    relu_cmplx,
    bnorm_cmplx,
    pool_cmplx,
    init_weights,
    max_mag_gabor_pool
)


log = logging.getLogger(__name__)


class IGaborCmplx(nn.Module):
    """Wraps the complex Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
        kernel_size (int, optional): Size of gabor kernel. Defaults to 3.
    """
    def __init__(self, no_g=4, layer=False, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.gabor_params = nn.Parameter(data=torch.Tensor(2, no_g))
        self.gabor_params.data[0] = torch.arange(no_g) / (no_g) * math.pi
        self.gabor_params.data[1].uniform_(-1 / math.sqrt(no_g),
                                           1 / math.sqrt(no_g))
        self.register_parameter(name="gabor", param=self.gabor_params)
        self.no_g = no_g
        self.register_buffer("gabor_filters", torch.Tensor(2, self.no_g, 1, 1,
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
        no_g (int, optional): The number of desired Gabor filters.
        gabor_pooling (str, optional): Type of pooling to apply across Gabor
            axis. Choices are [None, 'max', 'avg']. Defaults to None.
        include_gparams (bool, optional): Includes gabor params with highest
            activations as extra feature channels.
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 no_g=2, gabor_pooling=None, include_gparams=False,
                 weight_init=None, **conv_kwargs):
        if gabor_pooling is None:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        kernel_size = _pair(kernel_size)
        super().__init__()
        self.ReConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        self.ImConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        if weight_init is not None:
            init_weights(self.ReConv.weight, self.ImConv.weight, weight_init)
        self.conv = conv_cmplx

        self.gabor = IGaborCmplx(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.gabor_pooling = gabor_pooling
        self.include_gparams = include_gparams
        self.conv_kwargs = conv_kwargs

    def forward(self, x):
        enhanced_weight = self.gabor(cmplx(self.ReConv.weight, self.ImConv.weight))
        out = self.conv(x, enhanced_weight, **self.conv_kwargs)

        if self.gabor_pooling is None:
            return out

        pool_out = out.view(2,
                            out.size(1),
                            enhanced_weight.size(1) // self.no_g,
                            self.no_g,
                            out.size(3),
                            out.size(4))

        if self.gabor_pooling == 'max' or self.include_gparams:
            # print(out.min(), out.max(), out.mean())
            # pool_out, max_idxs = torch.max(pool_out, dim=3)
            pool_out, max_idxs = max_mag_gabor_pool(pool_out)
            # print(pool_out.min(), pool_out.max(), pool_out.mean())
            if self.gabor_pooling == 'max':
                out = pool_out
            if self.include_gparams:
                max_gparams = self.gabor.gabor_params[0, max_idxs]
                out = torch.stack(out, max_gparams, dim=3)

        if self.gabor_pooling == 'avg':
            out = torch.mean(pool_out, dim=3)

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
    def __init__(self, kernel_size, maxmag=True, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs
        self.operator = 'maxmag' if maxmag else None

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, operator=self.operator, **self.pool_kwargs)


class AvgPoolCmplx(nn.Module):
    """Implements complex average pooling.
    """
    def __init__(self, kernel_size, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, operator='avg', **self.pool_kwargs)
