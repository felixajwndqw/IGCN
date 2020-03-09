import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv
from igcn.cmplx_modules import (
    IGConvCmplx,
    ReLUCmplx,
    BatchNormCmplx,
    MaxPoolCmplx,
    MaxMagPoolCmplx,
    AvgPoolCmplx,
    ConvCmplx
)
from igcn.cmplx import new_cmplx, magnitude, concatenate


class DoubleIGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling='max',
                 no_g=4, prev_gabor_pooling=None, gabor_pooling=None, first=False):
        super().__init__()
        padding = kernel_size // 2
        first_div = 2 if first else 1
        max_g_div = no_g if gabor_pooling is not None else 1
        prev_max_g_div = no_g if gabor_pooling is not None else 1
        if 'max' in pooling:
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        self.double_conv = nn.Sequential(
            IGConv(
                in_channels // prev_max_g_div,
                out_channels // first_div,
                kernel_size,
                pooling=Pool,
                padding=padding,
                no_g=no_g,
                gabor_pooling=None
            ),
            IGConv(
                out_channels // first_div,
                out_channels // max_g_div,
                kernel_size,
                pooling=Pool,
                padding=padding,
                no_g=no_g,
                gabor_pooling=gabor_pooling,
                pool_kernel=2,
                pool_stride=2
            ),
            ReLUCmplx(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, prev_gabor_pooling=None, gabor_pooling=None,
                 pooling='mag', weight_init=None, all_gp=False,
                 relu_type='c', first=False, last=True):
        super().__init__()
        padding = kernel_size // 2 - 1
        all_gp_div = no_g if all_gp else 1
        max_g_div = no_g if gabor_pooling is not None else 1
        prev_max_g_div = no_g if prev_gabor_pooling is not None else 1
        first_div = 2 if first else 1
        all_gp = gabor_pooling if all_gp else None
        if pooling == 'max':
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        elif pooling == 'mag':
            Pool = MaxMagPoolCmplx
        self.double_conv = nn.Sequential(
            IGConvCmplx(
                in_channels // prev_max_g_div,
                out_channels // first_div // all_gp_div,
                kernel_size,
                padding=padding,
                no_g=no_g,
                gabor_pooling=all_gp,
                weight_init=weight_init
            ),
            IGConvCmplx(
                out_channels // first_div // all_gp_div,
                out_channels // max_g_div,
                kernel_size,
                padding=padding + int(last),
                no_g=no_g,
                gabor_pooling=gabor_pooling,
                weight_init=weight_init
            ),
            Pool(kernel_size=2, stride=2),
            BatchNormCmplx(out_channels // max_g_div),
            ReLUCmplx(
                inplace=True,
                relu_type=relu_type,
                channels=out_channels // max_g_div
            ),
        )

    def forward(self, x):
        return self.double_conv(x)


class SingleIGConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 no_g=4, prev_gabor_pooling=None, gabor_pooling=None,
                 pooling='mag', weight_init=None,
                 last=True, **kwargs):
        super().__init__()
        padding = kernel_size // 2 - 1
        max_g_div = no_g if gabor_pooling is not None else 1
        prev_max_g_div = no_g if prev_gabor_pooling is not None else 1
        if pooling == 'max':
            Pool = MaxPoolCmplx
        elif pooling == 'avg':
            Pool = AvgPoolCmplx
        if pooling == 'mag':
            Pool = MaxMagPoolCmplx
        self.double_conv = nn.Sequential(
            IGConvCmplx(
                in_channels // prev_max_g_div,
                out_channels // max_g_div,
                kernel_size,
                padding=padding,
                no_g=no_g,
                gabor_pooling=gabor_pooling,
                weight_init=weight_init
            ),
            Pool(kernel_size=2, stride=2),
            BatchNormCmplx(),
            ReLUCmplx(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class LinearBlock(nn.Module):
    def __init__(self, fcn, dropout=0., **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(fcn, fcn),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.block(x)


class LinearConvBlock(nn.Module):
    def __init__(self, channels, dropout=0., relu_type='c', **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            ConvCmplx(channels, channels, kernel_size=1),
            ReLUCmplx(inplace=True, relu_type=relu_type, channels=channels),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.block(x)


class IGCN(Model):
    """Model factory for IGCN.

    Args:
        n_classes (int, optional): Number of classes to estimate.
        n_channels (int, optional): Number of input channels.
        base_channels (int, optional): Number of feature channels in first layer of network.
        no_g (int, optional): The number of desired Gabor filters.
        kernel_size (int or tuple, optional): Size of kernel.
        inter_gp (str, optional): Type of pooling to apply across Gabor
            axis for intermediate layers.
            Choices are [None, 'max', 'avg']. Defaults to None.
        final_gp (str, optional): Type of pooling to apply across Gabor
            axis for the final layer.
            Choices are [None, 'max', 'avg']. Defaults to None.
        cmplx (bool, optional): Whether to use a complex architecture.
        pooling (str, optional): Type of pooling.
        dropout (float, optional): Probability of dropout layer(s).
        dset (str, optional): Type of dataset.
        single (bool, optional): Whether to use a single gconv layer between each pooling layer.
        all_gp (bool, optional): Whether to apply Gabor pooling on all layers.
        relu_type (str, optional): Type of relu layer. Choices are ['c', 'mod'].
            Defaults to 'c'.
        nfc (int, optional): Number of fully connected layers before classification.
        weight_init (str, optional): Type of weight initialisation.
        fc_type (str, optional): How complex tensors are combined into real
            tensors prior to FC. Choices are ['cat', 'mag']. Defaults to 'cat'.
    """
    def __init__(self, n_classes=10, n_channels=1, base_channels=16, no_g=4,
                 kernel_size=3, inter_gp=None, final_gp=None, cmplx=False,
                 pooling='max', dropout=0.3, dset='mnist', single=False,
                 all_gp=False, relu_type='c', nfc=2, weight_init=None,
                 fc_type='cat', fc_block='linear',
                 **kwargs):
        super().__init__(**kwargs)
        self.fc_type = fc_type
        if cmplx:
            ConvBlock = DoubleIGConvCmplx
            if single:
                ConvBlock = SingleIGConvCmplx
        else:
            ConvBlock = DoubleIGConv
        self.fc_block = fc_block
        if fc_block == 'lin':
            FCBlock = LinearBlock
        elif fc_block == 'cnv':
            FCBlock = LinearConvBlock
        self.conv1 = ConvBlock(
            n_channels,
            base_channels * 2,
            kernel_size,
            no_g=no_g,
            gabor_pooling=inter_gp,
            pooling=pooling,
            first=True,
            weight_init=weight_init,
            all_gp=all_gp,
            relu_type=relu_type
        )
        self.conv2 = ConvBlock(
            base_channels * 2,
            base_channels * 3,
            kernel_size,
            no_g=no_g,
            prev_gabor_pooling=inter_gp,
            gabor_pooling=inter_gp,
            pooling=pooling,
            weight_init=weight_init,
            all_gp=all_gp,
            relu_type=relu_type
        )
        self.conv3 = ConvBlock(
            base_channels * 3,
            base_channels * 4,
            kernel_size,
            no_g=no_g,
            prev_gabor_pooling=inter_gp,
            gabor_pooling=final_gp,
            pooling=pooling,
            last=True,
            weight_init=weight_init,
            all_gp=all_gp,
            relu_type=relu_type
        )
        # This line is hideous don't look at him
        self.fcn = 4 * base_channels // (no_g if final_gp else 1) * (4 if n_channels == 3 else 1)
        if cmplx and self.fc_block == 'lin' and self.fc_type == 'cat':
            self.fcn *= 2
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(FCBlock(self.fcn, dropout, relu_type=relu_type))
        self.linear = nn.Sequential(
            *linear_blocks,
        )
        if (self.fc_type == 'cat' and self.fc_block == 'cnv'):
            self.fcn *= 2
        self.classifier = nn.Sequential(
            nn.Linear(self.fcn, 10)
        )
        if self.fc_type == 'cat':
            self.project = concatenate
        elif self.fc_type == 'mag':
            self.project = magnitude
        self.cmplx = cmplx

    def forward(self, x):
        if self.cmplx:
            x = new_cmplx(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.fc_block == 'cnv':
            x = self.linear(x)
        if self.cmplx:
            x = self.project(x)
        x = x.flatten(1)
        if self.fc_block == 'lin':
            x = self.linear(x)
        x = self.classifier(x)
        return x
