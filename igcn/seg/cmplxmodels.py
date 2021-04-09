import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn.seg.cmplxigcn_unet_parts import DownCmplx, UpCmplx, TripleIGConvCmplx
from igcn.cmplx import new_cmplx, concatenate
from igcn.seg.scale import Scale


class UNetIGCNCmplx(Model):
    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16,
                 kernel_size=3, nfc=1, dropout=0., pooling='max',
                 mode='bilinear', gp='max', scale=False,
                 relu_type='mod', **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size

        if scale:
            self.preprocess = Scale(n_channels, method='arcsinh')
        else:
            self.preprocess = None
        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, first=True)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 4, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down3 = DownCmplx(base_channels * 4, base_channels * 8, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down4 = DownCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.up1 = UpCmplx(base_channels * 8, base_channels * 4, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up2 = UpCmplx(base_channels * 4, base_channels * 2, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up3 = UpCmplx(base_channels * 2, base_channels, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, no_g=no_g, gp=gp, relu_type=relu_type, mode=mode, last=True)

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
        if self.preprocess is not None:
            x = self.preprocess(x)
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
        x = concatenate(x)
        mask = self.outc(x)
        return mask
