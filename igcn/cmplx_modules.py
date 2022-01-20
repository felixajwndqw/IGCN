
import math
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from .gabor import gabor, gabor_cmplx, gabor_wavelet, norm, sin, GaborFunctionCmplx, GaborFunctionCyclicCmplx, GaborFunctionCmplxMult, GaborFunctionCyclicCmplxMult
from .utils import _pair
from .cmplx import (
    cmplx_mult,
    conv_cmplx,
    linear_cmplx,
    pool_cmplx,
    init_weights,
    max_mag_gabor_pool,
    max_summed_mag_gabor_pool,
    relu_cmplx,
    relu_cmplx_mod,
    relu_cmplx_z,
    magnitude,
    phase,
    concatenate
)
from igcn.utils import _compress_shape, _recover_shape


# log = logging.getLogger(__name__)


def cyclic_expand(t):
    no_g = t.size(2)
    cts = [t.roll(i, 2) for i in range(no_g)]
    ct = torch.stack(cts, dim=3)
    return ct


class IGaborCmplxManual(nn.Module):
    """Wraps the complex Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
        kernel_size (int, optional): Size of gabor kernel. Defaults to 3.
    """
    def __init__(self, no_g=4, kernel_size=3, cyclic=False, mod='hadam', **kwargs):
        super().__init__(**kwargs)
        self.theta = nn.Parameter(data=torch.Tensor(no_g))
        self.theta.data = torch.arange(no_g) / (no_g) * math.pi
        self.register_parameter(name="theta", param=self.theta)
        self.l = nn.Parameter(data=torch.Tensor(no_g))
        self.l.data.uniform_(
            -1 / math.sqrt(no_g),
            1 / math.sqrt(no_g)
        )
        self.register_parameter(name="lambda", param=self.l)
        self.no_g = no_g
        self.register_buffer(
            "gabor_filters",
            torch.Tensor(
                2,
                self.no_g,
                1,
                1,
                *kernel_size
            )
        )
        self.calc_filters = True  # Flag whether filter bank needs recalculating
        # self.register_backward_hook(self.set_filter_calc)
        self.cyclic = cyclic
        if mod == "hadam":
            if cyclic:
                self.GaborFunction = GaborFunctionCyclicCmplx.apply
            else:
                self.GaborFunction = GaborFunctionCmplx.apply
        elif mod == "cmplx":
            if cyclic:
                self.GaborFunction = GaborFunctionCyclicCmplxMult.apply
            else:
                self.GaborFunction = GaborFunctionCmplxMult.apply

    def forward(self, x):
        # if self.calc_filters:
        #     self.generate_gabor_filters(x)
        # print(f'x.size()={x.size()}, '
        #       f'x.unsqueeze(2).size()={x.unsqueeze(2).size()}, '
        #       f'gabor.size()={self.gabor_filters.size()}')
        # if self.cyclic:
        #     cyclic_gabor = cyclic_expand(self.gabor_filters)
        #     # print(f'x.size()={x.size()}, '
        #     #       f'x.unsqueeze(2).unsqueeze(2).size()={x.unsqueeze(2).unsqueeze(2).size()}, '
        #     #       f'gabor.unsqueeze(4).size()={cyclic_gabor.size()}, '
        #     #       f'gabor.size()={cyclic_gabor.size()}')
        #     out = cyclic_gabor * x.unsqueeze(2).unsqueeze(2)
        # else:
        #     out = self.gabor_filters * x.unsqueeze(2)

        # print(f'self.gabor_filters.size()={self.gabor_filters.size()}, x.unsqueeze(2).size()={x.unsqueeze(2).size()}')
        # print(f'x.unsqueeze(2).size()={x.unsqueeze(2).size()}')
        out = self.GaborFunction(
            x.unsqueeze(2),
            torch.stack((
                self.theta,
                self.l
            ))
        )

        # out = out.view(2, -1 , *out.size()[3:])
        # print(f'gabor_out.size()={out.size()}')
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        self.gabor_filters = gabor_cmplx(
            x,
            torch.stack((
                self.theta,
                self.l
            ))
        ).unsqueeze(1)
        self.calc_filters = False

    def set_filter_calc(self, *args):
        """Called in backward hook so that filter bank will be regenerated.
        """
        self.calc_filters = True


class IGaborCmplx(nn.Module):
    """Wraps the complex Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
        kernel_size (int, optional): Size of gabor kernel. Defaults to 3.
        single_param (boolean, optional): Uses single lambda/sigma for gabor filters,
            rather than no_g different variables.
    """
    def __init__(self, no_g=4, kernel_size=3, cyclic=False, mod='cmplx',
                 l_init='uniform', sigma_init='fixed', single_param=False,
                 morlet=False, **kwargs):
        super().__init__(**kwargs)
        self.no_g = no_g
        self.morlet = morlet
        self.theta = nn.Parameter(data=torch.Tensor(no_g))
        self.theta.data = torch.arange(no_g, dtype=torch.float) / (no_g) * math.pi
        self.register_parameter(name="theta", param=self.theta)
        if not single_param:
            self.l = nn.Parameter(data=torch.Tensor(no_g))
            self.sigma = nn.Parameter(data=torch.Tensor(no_g))
        else:
            self.l = nn.Parameter(data=torch.Tensor(1))
            self.sigma = nn.Parameter(data=torch.Tensor(1))
        self.init_lambda(l_init)
        self.init_sigma(sigma_init)
        self.register_parameter(name="lambda", param=self.l)
        self.data_dim = '2d'

        if type(kernel_size) is tuple:
            if len(kernel_size) == 3:
                self.data_dim = '3d'
                kernel_size = kernel_size[1:]

        if cyclic:
            self.register_buffer(
                "gabor_filters",
                torch.Tensor(
                    2,
                    1,
                    self.no_g,
                    self.no_g,
                    1,
                    *kernel_size
                )
            )
        else:
            self.register_buffer(
                "gabor_filters",
                torch.Tensor(
                    2,
                    1,
                    self.no_g,
                    1,
                    *kernel_size
                )
            )

        self.calc_filters = True  # Flag whether filter bank needs recalculating
        self.register_full_backward_hook(self.set_filter_calc)
        self.cyclic = cyclic
        if mod == "hadam":
            self.modulate = torch.multiply
        elif mod == "cmplx":
            self.modulate = cmplx_mult

        self.morlet = morlet
        if morlet:
            self.filter = gabor_wavelet
        else:
            self.filter = gabor_cmplx

    def forward(self, x):
        if self.calc_filters:
            self.generate_gabor_filters(x)
        # print(f'x.size()={x.size()}, '
        #       f'x.unsqueeze(2).size()={x.unsqueeze(2).size()}, '
        #       f'gabor.size()={self.gabor_filters.size()}')

        if self.cyclic:
            x = x.unsqueeze(2)
        out = self.modulate(self.gabor_filters, x.unsqueeze(2))

        # print(f'self.gabor_filters.size()={self.gabor_filters.size()}, x.unsqueeze(2).size()={x.unsqueeze(2).size()}')
        # print(f'x.unsqueeze(2).size()={x.unsqueeze(2).size()}')

        # out = out.view(2, -1 , *out.size()[3:])
        # print(f'gabor_out.size()={out.size()}')
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        filt = self.filter(
            x,
            self.theta,
            self.l,
            self.sigma
        ).unsqueeze(1)
        if self.cyclic:
            filt = cyclic_expand(filt)
        if self.data_dim == '3d':
            filt = filt.unsqueeze(-3)
        self.gabor_filters = filt
        self.calc_filters = False

    def set_filter_calc(self, *args):
        """Called in backward hook so that filter bank will be regenerated.
        """
        self.calc_filters = True

    def init_lambda(self, l_init, mean=3):
        if l_init == 'uniform':
            self.l.data.uniform_(
                1.5,
                3
            )
        elif l_init == 'normalg':
            self.l.data.normal_(mean / math.sqrt(self.no_g), std=.25 * math.sqrt(self.no_g))
        elif l_init == 'normal':
            self.l.data.normal_(mean / math.sqrt(2), std=.25)
        elif l_init == 'fixed':
            self.l.data.fill_(mean)
            self.l.requires_grad = False
        else:
            raise ValueError("Unknown lambda initialisation type: " + l_init)

    def init_sigma(self, sigma_init, mean=math.pi):
        if sigma_init == 'normal':
            self.sigma.data.normal_(mean, std=.25)
        elif sigma_init == 'fixed':
            self.sigma.data.fill_(mean)
            self.sigma.requires_grad = False
        else:
            raise ValueError("Unknown lambda initialisation type: " + sigma_init)


class IGaborCmplx2(nn.Module):
    """Wraps the complex Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
        kernel_size (int, optional): Size of gabor kernel. Defaults to 3.
    """
    def __init__(self, no_g=4, kernel_size=3, cyclic=False, **kwargs):
        super().__init__(**kwargs)
        self.theta = nn.Parameter(data=torch.Tensor(no_g))
        self.theta.data = torch.arange(no_g) / (no_g) * math.pi
        self.register_parameter(name="theta", param=self.theta)
        self.no_g = no_g
        self.register_buffer(
            "gabor_filters",
            torch.Tensor(
                2,
                self.no_g,
                1,
                1,
                *kernel_size
            )
        )
        self.calc_filters = True  # Flag whether filter bank needs recalculating
        self.register_backward_hook(self.set_filter_calc)
        self.cyclic = cyclic
        # if cyclic:
        #     self.GaborFunction = GaborFunctionCmplx.apply
        # else:
        #     self.GaborFunction = GaborFunctionCyclicCmplx.apply

    def forward(self, x):
        if self.calc_filters:
            self.generate_gabor_filters(x)
        # print(f'x.size()={x.size()}, '
        #       f'x.unsqueeze(2).size()={x.unsqueeze(2).size()}, '
        #       f'gabor.size()={self.gabor_filters.size()}')
        if self.cyclic:
            cyclic_gabor = cyclic_expand(self.gabor_filters)
            # print(f'x.size()={x.size()}, '
            #       f'x.unsqueeze(2).unsqueeze(2).size()={x.unsqueeze(2).unsqueeze(2).size()}, '
            #       f'gabor.unsqueeze(4).size()={cyclic_gabor.size()}, '
            #       f'gabor.size()={cyclic_gabor.size()}')
            out = cyclic_gabor * x.unsqueeze(2).unsqueeze(2)
        else:
            out = self.gabor_filters * x.unsqueeze(2)

        # out = out.view(2, -1 , *out.size()[3:])
        # print(f'out.size()={out.size()}')
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        self.gabor_filters = sin(
            x,
            self.theta
        ).unsqueeze(1)
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
                 weight_init='he', mod='hadam',
                 l_init='uniform', sigma_init='fixed', single_param=False, morlet=False,
                 **conv_kwargs):
        kernel_size = _pair(kernel_size)
        self.kernel_size = kernel_size
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(
            2,
            output_features,
            input_features,
            *kernel_size
        ))
        if weight_init is not None:
            init_weights(self.weight, weight_init)
        self.conv = conv_cmplx
        self.dim = len(kernel_size)

        self.gabor = IGaborCmplx(no_g, kernel_size=kernel_size, mod=mod,
                                 l_init=l_init, sigma_init=sigma_init, single_param=single_param, morlet=morlet)
        self.no_g = no_g
        self.gabor_pooling = GaborPool(gabor_pooling) if gabor_pooling is not None else None

        self.include_gparams = include_gparams
        self.conv_kwargs = sanitise_conv_kwargs(conv_kwargs)

    def forward(self, x):
        tw = self.gabor(self.weight)
        # print(f'x.size()={x.size()}, tw.size()={tw.size()}, self.weight.size()={self.weight.size()}')
        if x.dim() == 4 + self.dim:
            x = x.view(
                2,
                x.size(1),
                x.size(2) * x.size(3),
                *x.size()[4:]
            )
        tw = tw.view(
            2,
            tw.size(1) * self.no_g,
            *tw.size()[3:]
        )
        # print(f'x.size()={x.size()}, tw.size()={tw.size()}, self.weight.size()={self.weight.size()}')
        out = self.conv(x, tw, **self.conv_kwargs)
        # print(f'out.size()={out.size()}')
        out = out.view(
            2,
            out.size(1),
            out.size(2) // self.no_g,
            self.no_g,
            *out.size()[3:]
        )

        if self.gabor_pooling is None and self.include_gparams is False:
            return out

        # pool_out, max_idxs = self.gabor_pooling(out, dim=3)
        # pool_out = pool_out.unsqueeze(3)
        pool_out = self.gabor_pooling(out)
        if self.include_gparams:
            max_thetas = self.gabor.theta[max_idxs]
            return out, max_thetas[0]  # Just returns real thetas in complex
        # print(f'pool_out={pool_out.unsqueeze(3).size()}')
        return pool_out.unsqueeze(3)


class IGConvGroupCmplx(nn.Module):
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
                 weight_init='he', mod='hadam',
                 l_init='uniform', sigma_init='fixed', single_param=False, morlet=False,
                 **conv_kwargs):
        kernel_size = _pair(kernel_size)
        self.kernel_size = kernel_size
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(
            2,
            output_features,
            input_features,
            *kernel_size
        ))
        if weight_init is not None:
            init_weights(self.weight, weight_init)
        self.conv = conv_cmplx
        self.dim = len(kernel_size)
        conv_kwargs['data_dim'] = f'{self.dim}d'

        self.gabor = IGaborCmplx(no_g, kernel_size=kernel_size, cyclic=True, mod=mod,
                                 l_init=l_init, sigma_init=sigma_init, single_param=single_param, morlet=morlet)
        self.no_g = no_g
        self.gabor_pooling = GaborPool(gabor_pooling) if gabor_pooling is not None else None

        self.include_gparams = include_gparams
        self.conv_kwargs = sanitise_conv_kwargs(conv_kwargs)

    def forward(self, x):
        # print(f'self.weight.size()={self.weight.size()}')
        tw = self.gabor(self.weight)
        # print(f'x.size()={x.size()}, tw.size()={tw.size()}')
        x = x.view(
            2,
            x.size(1),
            self.input_features * x.size(3),
            *x.size()[4:]
        )
        tw = tw.view(
            2,
            self.output_features * self.no_g,
            self.input_features * self.no_g,
            *tw.size()[5:]
        )
        # print(f'x.size()={x.size()}, tw.size()={tw.size()}')
        out = self.conv(x, tw, **self.conv_kwargs)
        # print(f'out.size()={out.size()}')
        out = out.view(
            2,
            out.size(1),
            self.output_features,
            self.no_g,
            *out.size()[3:]
        )

        if self.gabor_pooling is None and self.include_gparams is False:
            return out

        # pool_out, max_idxs = self.gabor_pooling(out, dim=3)
        # pool_out = pool_out.unsqueeze(3)
        # print(f'pool_out={out.size()}')
        # pool_out, max_idxs = self.gabor_pooling(out, dim=3)
        pool_out = self.gabor_pooling(out)
        if self.include_gparams:
            max_thetas = self.gabor.theta[max_idxs]
            return out, max_thetas[0]  # Just returns real thetas in complex
        return pool_out#.unsqueeze(3)


class Project(nn.Module):
    """Projects a complex layer to real
    """
    def __init__(self, projection=None):
        super().__init__()
        self.projection = projection
        if self.projection in ('cat', 'bmp', 'nmp'):
            self.mult = 2
        else:
            self.mult = 1

    def forward(self, x):
        if self.projection is None:
            return x
        if self.projection == 'mag':
            x = magnitude(x)
        if self.projection == 'cat':
            x = concatenate(x)
        if self.projection == 'bmp':
            mags = magnitude(x)
            phases = phase(x)
            x = torch.stack([mags, phases], dim=0)
        if self.projection == 'nmp':
            mags = norm(magnitude(x))
            phases = norm(phase(x))
            x = torch.stack([mags, phases], dim=0)
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
                 weight_init='he', **conv_kwargs):
        kernel_size = _pair(kernel_size)
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(
            2,
            output_features,
            input_features,
            *kernel_size
        ))
        if weight_init is not None:
            init_weights(self.weight, weight_init)
        self.conv = conv_cmplx
        self.conv_kwargs = sanitise_conv_kwargs(conv_kwargs)

    def forward(self, x):
        out = self.conv(x, self.weight, **self.conv_kwargs)
        return out


class LinearCmplx(nn.Module):
    """Implements a complex linear layer.

    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        bias (bool, optional): Whether to use biases. Defaults to True.
    """
    def __init__(self, input_features, output_features, bias=True,
                 weight_init='he'):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(
            2,
            output_features,
            input_features
        ))
        init_weights(self.weight, weight_init)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                2,
                output_features
            ))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        out = linear_cmplx(x, self.weight, self.bias)
        return out


class LinearMagPhase(nn.Module):
    """Implements parallel linear layers for mags and phases.

    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        bias (bool, optional): Whether to use biases. Defaults to True.
    """
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.MagLinear = nn.Linear(input_features, output_features, bias=bias)
        self.PhaseLinear = nn.Linear(input_features, output_features, bias=bias)
        self.bias = bias

    def forward(self, x):
        return torch.stack([self.MagLinear(x[0]), self.PhaseLinear(x[1])], dim=0)


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
        self.relu_type = relu_type
        if relu_type == 'c':
            self.relu = relu_cmplx
        elif relu_type == 'z':
            self.relu = relu_cmplx_z
        elif relu_type == 'mod':
            assert channels is not None
            self.b = nn.Parameter(data=torch.Tensor(1, channels, 1, 1))
            init.zeros_(self.b)
            self.register_parameter(name="b", param=self.b)
            self.relu_kwargs['b'] = self.b
            self.relu = relu_cmplx_mod
        elif relu_type == 'mf':
            assert channels is not None
            self.b = nn.Parameter(data=torch.Tensor(1, channels, 1, 1))
            init.zeros_(self.b)
            self.register_parameter(name="b", param=self.b)
            self.b.requires_grad = False
            self.relu_kwargs['b'] = self.b
            self.relu = relu_cmplx_mod

    def forward(self, x):
        return self.relu(x, **self.relu_kwargs)


class BatchNormCmplxOld(nn.Module):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.size(1) == 1:
            return x

        x, xs = _compress_shape(x)
        spatial_dims = torch.tensor(range(x.dim())[3:])

        means = torch.mean(x, (1, *spatial_dims), keepdim=True)
        x = x - means

        stds = torch.std(magnitude(x, eps=self.eps, sq=False), (0, *spatial_dims - 1), keepdim=True)
        x = x / torch.clamp(stds.unsqueeze(0), min=self.eps)

        x = _recover_shape(x, xs)
        return x


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


class GaborPool(nn.Module):
    """Implements pooling over Gabor axis.
    """
    def __init__(self, pool_type='max', no_g=None, **pool_kwargs):
        super().__init__()
        self.no_g = no_g
        self.pool_type = pool_type
        if pool_type == 'max':
            self.pooling = torch.max
        elif pool_type == 'avg':
            self.pooling = torch.mean
        elif pool_type == 'mag':
            self.pooling = max_mag_gabor_pool
        elif pool_type == 'sum':
            self.pooling = max_summed_mag_gabor_pool
        elif pool_type is None:
            self.pooling = Identity()

    def forward(self, x):
        if self.no_g is not None:
            _, b, c, w, h = x.shape
            x = x.view(2, b, c // self.no_g, self.no_g, w, h)
        x = self.pooling(x, dim=3)
        if type(x) is tuple or self.pool_type == 'max':
            x = x[0]
        return x


class Identity(nn.Module):
    """Argument agnostic identity layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ReshapeGabor(nn.Module):
    def __init__(self, gabor_dim=2):
        super().__init__()
        self.gabor_dim = gabor_dim

    def forward(self, x):
        x, _ = _compress_shape(x)
        return x


def sanitise_conv_kwargs(conv_kwargs):
    valid_keys = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
    for key in list(conv_kwargs.keys()):
        if key not in valid_keys:
            conv_kwargs.pop(key)
    return conv_kwargs
