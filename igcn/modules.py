import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd, Conv2d
from .gabor import GaborFunction, gabor
# from .vis import FilterPlot
from .rot_pool import RotMaxPool2d
from .utils import _pair


class IGabor(nn.Module):
    """Wraps the Gabor implementation into a NN layer w/o convolution.

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
        self.GaborFunction = GaborFunction.apply
        self.register_buffer("gabor_filters", torch.Tensor(no_g, 1, 1,
                                                           *kernel_size))

        self.no_g = no_g
        self.layer = layer
        self.calc_filters = True  # Flag whether filter bank needs recalculating
        self.register_full_backward_hook(self.set_filter_calc)

    def forward(self, x):
        # print(f'x.size()={x.unsqueeze(1).size()}, gabor={gabor(x, self.gabor_params).unsqueeze(1).size()}')
        if self.calc_filters:
            self.generate_gabor_filters(x)

        # print(f'self.gabor_filters.size()={self.gabor_filters.size()}')

        out = self.gabor_filters * x.unsqueeze(1)

        # print(f'out.size()={out.size()}')

        out = out.view(-1 , *out.size()[2:])

        # print(f'out.size()={out.size()}')

        if self.layer:
            out = out.view(x.size(0), x.size(1) * self.no_g, *x.size()[2:])
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        self.gabor_filters = gabor(x, self.gabor_params).unsqueeze(1)
        self.calc_filters = False

    def set_filter_calc(self, *args):
        """Called in backward hook so that filter bank will be regenerated.
        """
        self.calc_filters = True


class IGConv(Conv2d):
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
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 pooling=None, no_g=2, pool_stride=1, pool_kernel=3, plot=False,
                 max_gabor=False, include_gparams=False, **conv_kwargs):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        kernel_size = _pair(kernel_size)
        super().__init__(input_features, output_features, kernel_size, **conv_kwargs)
        self.conv = F.conv2d

        self.gabor = IGabor(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.pooling = None
        if pooling is not None:
            self.pooling = pooling(kernel_size=pool_kernel, stride=pool_stride)
        self.max_gabor = max_gabor
        self.conv_kwargs = conv_kwargs
        self.bn = nn.BatchNorm2d(output_features * no_g)
        self.include_gparams = include_gparams

        self.plot = None

    def forward(self, x):
        enhanced_weight = self.gabor(self.weight)
        out = self.conv(x, enhanced_weight, **self.conv_kwargs)
        out = self.bn(out)

        if self.plot is not None:
            self.plot.update(self.weight[:, 0].clone().detach().cpu().numpy(),
                             enhanced_weight[:, 0].clone().detach().cpu().numpy(),
                             self.gabor_params.clone().detach().cpu().numpy())

        if self.pooling is not None:
            out = self.pooling(out)

        if self.max_gabor or self.include_gparams:
            max_out = out.view(out.size(0),
                               enhanced_weight.size(0) // self.no_g,
                               self.no_g,
                               out.size(2),
                               out.size(3))
            max_out, max_idxs = torch.max(max_out, dim=2)

        if self.max_gabor:
            out = max_out

        if self.include_gparams:
            max_gparams = self.gabor.gabor_params[0, max_idxs]
            out = torch.stack(out, max_gparams, dim=3)

        return out


class IGBranched(Conv2d):
    """Concatenates Gabor filtered input onto activation maps from convolution.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 rot_pool=None, no_g=2, pool_stride=1, plot=False,
                 max_gabor=False, **conv_kwargs):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features = (output_features - no_g) // input_features
        kernel_size = _pair(kernel_size)
        super().__init__(input_features, output_features, kernel_size, **conv_kwargs)
        self.conv = F.conv2d

        self.gabor = IGabor(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.rot_pool = rot_pool
        self.max_gabor = max_gabor
        self.conv_kwargs = conv_kwargs
        print(f'stride={self.stride}, padding={self.padding}, dilation={self.dilation}')

    def forward(self, x):
        gabor_out = self.gabor(x)
        print(gabor_out.size())
        gabor_out = gabor_out.view(x.size(0), x.size(1) * self.no_g, *x.size()[2:])
        print(gabor_out.size())
        conv_out = self.conv(x, self.weight, **self.conv_kwargs)
        print(x.size(), self.weight.size(), gabor_out.size(), conv_out.size())
        return torch.cat((gabor_out, conv_out), dim=1)


class IGParallel(_ConvNd):
    """Concatenates Gabor modulated conv onto activation maps from convolution.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 stride=1, padding=0, dilation=1, bias=None,
                 rot_pool=False, no_g=2, pool_stride=1, plot=False,
                 max_gabor=True):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        else:
            output_features //= 2
        kernel_size = _pair(kernel_size)
        super().__init__(
            input_features, output_features, kernel_size,
            stride, padding, dilation, False, (0, 0), 1, bias, 'zeros'
        )
        self.gabor = IGabor(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.max_gabor = max_gabor

    def forward(self, x):
        enhanced_weight = self.gabor(self.weight)
        gabor_conv_out = F.conv2d(x, enhanced_weight, None, self.stride,
                                  self.padding, self.dilation)
        if self.max_gabor:
            gabor_conv_out = gabor_conv_out.view(gabor_conv_out.size(0),
                                                 enhanced_weight.size(0) // self.no_g,
                                                 self.no_g,
                                                 gabor_conv_out.size(2),
                                                 gabor_conv_out.size(3))
            gabor_conv_out, _ = torch.max(gabor_conv_out, dim=2)
        conv_out = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation)
        return torch.cat((gabor_conv_out, conv_out), dim=1)


class MaxGabor(nn.Module):
    """
    """
    def __init__(self, no_g, **kwargs):
        super().__init__(**kwargs)
        self.no_g = no_g

    def forward(self, x):
        reshaped = x.view(x.size(0), x.size(1) // self.no_g, self.no_g, x.size(2), x.size(3))
        _, max_idxs = torch.max(reshaped, dim=2)
        return torch.cat((x, max_idxs.float()), dim=1)
