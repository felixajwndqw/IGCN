
import math
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.conv import Conv2d
from .igabor import gabor_cmplx
from .utils import _pair
from .cmplx import (
    cmplx,
    conv_cmplx,
    linear_cmplx,
    pool_cmplx,
    init_weights,
    max_mag_gabor_pool,
    max_summed_mag_gabor_pool,
    relu_cmplx,
    relu_cmplx_mod,
    relu_cmplx_z,
    bnorm_cmplx_old,
    magnitude,
    concatenate
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
            axis. Choices are [None, 'max', 'mag', 'avg']. Defaults to None.
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
        if gabor_pooling == 'max':
            gabor_pooling = torch.max
        elif gabor_pooling == 'avg':
            gabor_pooling = lambda x, dim: (torch.mean(x, dim=dim), None)
        elif gabor_pooling == 'mag':
            gabor_pooling = max_mag_gabor_pool
        elif gabor_pooling == 'sum':
            gabor_pooling = max_summed_mag_gabor_pool

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

        pool_out, max_idxs = self.gabor_pooling(pool_out, dim=3)
        out = pool_out


        # if self.include_gparams:
            # if self.gabor_pooling == 'max':
            #     out = pool_out
            # if self.include_gparams:
            #     max_gparams = self.gabor.gabor_params[0, max_idxs]
            #     out = torch.stack(out, max_gparams, dim=3)

        return out


class Project(nn.Module):
    """Projects a complex layer to real
    """
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, x):
        if self.projection == 'mag':
            x = magnitude(x)
        if self.projection == 'cat':
            x = concatenate(x)
        return x


class ConvCmplx(nn.Module):
    """Implements a complex convolutional layer.

    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        kernel_size (int, tuple): Size of kernel.
        weight_init (str, optional): Type of weight initialisation method.
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 weight_init=None, **conv_kwargs):
        kernel_size = _pair(kernel_size)
        super().__init__()
        self.ReConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        self.ImConv = Conv2d(input_features, output_features, kernel_size, **conv_kwargs)
        if weight_init is not None:
            init_weights(self.ReConv.weight, self.ImConv.weight, weight_init)
        self.conv = conv_cmplx
        self.conv_kwargs = conv_kwargs

    def forward(self, x):
        cmplx_weight = cmplx(self.ReConv.weight, self.ImConv.weight)
        out = self.conv(x, cmplx_weight, **self.conv_kwargs)
        return out


class LinearCmplx(nn.Module):
    """Implements a complex linear layer.

    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        bias (bool, optional): Whether to use biases. Defaults to True.
    """
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.ReLinear = nn.Linear(input_features, output_features, bias=bias)
        self.ImLinear = nn.Linear(input_features, output_features, bias=bias)
        self.bias = bias
        self.linear = linear_cmplx

    def forward(self, x):
        cmplx_weight = cmplx(self.ReLinear.weight, self.ImLinear.weight)
        if self.bias:
            cmplx_bias = cmplx(self.ReLinear.bias, self.ImLinear.bias)
        else:
            cmplx_bias = None
        out = self.linear(x, cmplx_weight, cmplx_bias)
        return out


class ReLUCmplx(nn.Module):
    """Implements complex rectified linear unit.

    if relu_type == 'c':
        x' = relu(re(x)) + i*relu(im(x))
    if relu_type == 'z':
        x' = x if both re(x) and im(x) are positive <=> 0<arctan(im(x)/re(x))<pi/2
    if relu_type == 'mod':
        x' = relu(|x|+b)*(x/|x|)

        Biases are pulled from uniformly from between [1/sqrt(c),-1/sqrt(c)]
    """
    def __init__(self, inplace=False, relu_type='c', channels=None):
        super().__init__()
        self.relu_kwargs = {'inplace': inplace}
        if relu_type == 'c':
            self.relu = relu_cmplx
        elif relu_type == 'z':
            self.relu = relu_cmplx_z
        elif relu_type == 'mod':
            assert channels is not None
            self.b = nn.Parameter(data=torch.Tensor(channels, 1, 1))
            init.zeros_(self.b)
            # self.b.data.uniform_(-1 / math.sqrt(channels),
            #                      1 / math.sqrt(channels))
            self.register_parameter(name="b", param=self.b)
            # self.b.requires_grad = False
            self.relu_kwargs['b'] = self.b
            self.relu = relu_cmplx_mod

    def forward(self, x):
        return self.relu(x, **self.relu_kwargs)


class BatchNormCmplx(nn.Module):
    """Implements complex batch normalisation.
    """
    def __init__(self, num_features, momentum=0.999, eps=1e-8, bnorm_type='new'):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1, 1))
        self.weight = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.bnorm_type = bnorm_type
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.bnorm_type == 'old':
            return bnorm_cmplx_old(x, self.eps)
        r = magnitude(x, eps=self.eps)
        batch_mean = r.mean((0, 2, 3), keepdim=True)
        batch_var = r.var((0, 2, 3), keepdim=True)

        mean = (1.0 - self.momentum) * batch_mean + self.momentum * self.running_mean
        var = (1.0 - self.momentum) * batch_var + self.momentum * self.running_var

        r_bn = self.weight * (r - mean) / (var + self.eps) + self.bias
        return r_bn * x / r


class MaxMagPoolCmplx(nn.Module):
    """Implements complex max by magnitude pooling.
    """
    def __init__(self, kernel_size, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, operator='mag', **self.pool_kwargs)


class MaxPoolCmplx(nn.Module):
    """Implements complex max pooling.
    """
    def __init__(self, kernel_size, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, operator='max', **self.pool_kwargs)


class AvgPoolCmplx(nn.Module):
    """Implements complex average pooling.
    """
    def __init__(self, kernel_size, **pool_kwargs):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.pool_kwargs = pool_kwargs

    def forward(self, x):
        return pool_cmplx(x, self.kernel_size, operator='avg', **self.pool_kwargs)
