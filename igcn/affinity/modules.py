import torch
import torch.nn as nn

from igcn.cmplx_modules import AvgPoolCmplx, MaxMagPoolCmplx, MaxPoolCmplx
from igcn.cmplx import cmplx
from igcn.utils import _compress_shape, _recover_shape, _trio
from igcn.seg.cmplx_modules import TripleIGConvCmplx


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, no_g=4, pooling='avg', gp='max', relu_type='mod', **kwargs):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(scale_factor, data_dim='3d')
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx(scale_factor, data_dim='3d')
        else:
            Pool = MaxPoolCmplx(scale_factor, data_dim='3d')
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, _trio(kernel_size), no_g=no_g, gp=gp, relu_type=relu_type, data_dim='3d', **kwargs)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels:
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, no_g=4, mode='nearest', last=False, gp=None, relu_type='mod', comb_fn=torch.add, **kwargs):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=tuple(scale_factor), mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, _trio(kernel_size), no_g=no_g, last=last, gp=gp, relu_type=relu_type, data_dim='3d', **kwargs)
        self.comb_fn = comb_fn

    def forward(self, x1, x2):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(self.comb_fn(x1, x2))
