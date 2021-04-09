import torch
import torch.nn as nn
from igcn.cmplx import cmplx
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx_modules import IGConvCmplx, IGConvGroupCmplx, ReLUCmplx, MaxPoolCmplx, AvgPoolCmplx, MaxMagPoolCmplx, BatchNormCmplxOld
from igcn.utils import _compress_shape, _recover_shape


class TripleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, include_gparams=False, last=False, gp='max',
                 first=False):
        super().__init__()
        first_conv = IGConvGroupCmplx
        if first:
            first_conv = IGConvCmplx
        padding = kernel_size // 2

        self.conv1 = first_conv(
            in_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,
            padding=padding, gabor_pooling=None)
        self.bn_relu1 = nn.Sequential(
            # BatchNormCmplx(out_channels * no_g, bnorm_type='new'),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )

        self.conv2 = IGConvGroupCmplx(
            out_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,
            padding=padding, gabor_pooling=None)
        self.bn_relu2 = nn.Sequential(
            # BatchNormCmplx(out_channels * no_g, bnorm_type='new'),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )

        self.conv3 = IGConvGroupCmplx(
            out_channels, out_channels, no_g=no_g,
            kernel_size=kernel_size,
            padding=padding, gabor_pooling=gp,
            include_gparams=include_gparams)
        self.bn_relu3 = nn.Sequential(
            # BatchNormCmplx(out_channels * (no_g if gp is None else 1), bnorm_type='new'),
            BatchNormCmplxOld(),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )

        self.include_gparams = include_gparams

        # if include_gparams:
        #     self.handle_last = self.extract_theta
        # else:
        #     self.handle_last = self.normal_conv

    # def extract_theta(self, x):
    #     x, t = self.conv3(x)
    #     # print(f'x.size()={x.size()}, t.type()={t.type()}, self.include_gparams={self.include_gparams}')
    #     return self.bn_relu3(x), t

    # def normal_conv(self, x):
    #     return self.bn_relu3(self.conv3(x))

    def forward(self, x):
        # print(f'x.size()={x.size()}')
        x1 = self.bn_relu1(self.conv1(x))
        # print(f'x1.size()={x1.size()}')
        x2 = self.bn_relu2(self.conv2(x1))
        # print(f'x2.size()={x2.size()}')
        return self.bn_relu3(self.conv3(x2))
        # return self.handle_last(x1 + x2)


    # def extract_theta(self):
    #     print(self.conv1[0].gabor.gabor_params[0])
    #     t = torch.mean(
    #         torch.stack((
    #             self.conv1[0].gabor.gabor_params[0],
    #             self.conv2[0].gabor.gabor_params[0],
    #             self.conv3[0].gabor.gabor_params[0]
    #         ))
    #     )
    #     return t


class DownCmplx(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, pooling='avg', gp='max'):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, gp=gp)
        )

    def forward(self, x):
        return self.pool_conv(x)


class DownCmplxAngle(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, pooling='avg', gp='max'):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, gp=gp, include_gparams=True)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpCmplx(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, mode='nearest', last=False, gp=None):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, last=last, gp=gp)

    def forward(self, x1, x2):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(x1 + x2)


class UpSimpleCmplx(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, no_g=4, mode='nearest', last=False, gp=None):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleIGConvCmplx(in_channels, out_channels, kernel_size, no_g=no_g, last=last, gp=gp)

    def forward(self, x1):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(x1)
