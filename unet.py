import torch
import torch.nn as nn
import torch.nn.functional as F
from quicktorch.models import Model
from igcn.cmplx import new_cmplx, cmplx
from igcn.cmplx_modules import ConvCmplx, BatchNormCmplxOld, ReLUCmplx, MaxPoolCmplx, AvgPoolCmplx
from igcn.cmplx_bn import BatchNormCmplx


class UNetCmplx(Model):
    def __init__(self, n_classes, n_channels=1, base_channels=16,
                 kernel_size=3, nfc=1, dropout=0., pooling='max', mode='nearest', **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size

        self.inc = TripleConvCmplx(n_channels, base_channels, kernel_size)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size, pooling=pooling)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 4, kernel_size, pooling=pooling)
        self.down3 = DownCmplx(base_channels * 4, base_channels * 8, kernel_size, pooling=pooling)
        self.down4 = DownCmplx(base_channels * 8, base_channels * 8, kernel_size, pooling=pooling)
        self.up1 = UpCmplx(base_channels * 8, base_channels * 4, kernel_size, mode=mode)
        self.up2 = UpCmplx(base_channels * 4, base_channels * 2, kernel_size, mode=mode)
        self.up3 = UpCmplx(base_channels * 2, base_channels, kernel_size, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, mode=mode)

        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
        self.outc = nn.Sequential(
            *linear_blocks,
            nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = new_cmplx(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x[0], x[1]], dim=1)
        mask = self.outc(x)
        return mask


class TripleConvCmplx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            ConvCmplx(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )
        self.conv2 = nn.Sequential(
            ConvCmplx(out_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )
        self.conv3 = nn.Sequential(
            ConvCmplx(out_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding),
            BatchNormCmplx(out_channels),
            ReLUCmplx(relu_type='mod', channels=out_channels, inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1 + x2)
        return x3


class DownCmplx(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='avg'):
        super().__init__()
        if pooling == 'avg':
            Pool = AvgPoolCmplx(2)
        else:
            Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            TripleConvCmplx(in_channels, out_channels, kernel_size)
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

    def __init__(self, in_channels, out_channels, kernel_size=3, mode='nearest'):
        super().__init__()

        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = TripleConvCmplx(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))
        return self.conv(x1 + x2)
