import torch
import torch.nn as nn
from quicktorch.models import Model
from igcn.cmplx import new_cmplx, concatenate
from igcn.cmplx_modules import IGConvCmplx, MaxPoolCmplx, AvgPoolCmplx, MaxMagPoolCmplx
from igcn.seg.modules import Down, Up, TripleIGConv
from igcn.seg.cmplx_modules import DownCmplx, UpCmplx, TripleIGConvCmplx, RCFBlock
from igcn.seg.scale import Scale, ScaleParallel


class UNetIGCNCmplx(Model):
    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16,
                 kernel_size=3, nfc=1, dropout=0., pooling='max',
                 mode='bilinear', gp='max', scale=False,
                 relu_type='mod', pad_to_remove=64, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.gp = gp
        self.p = pad_to_remove // 2

        in_channels_conv = n_channels

        if scale == 'parallel':
            self.preprocess = ScaleParallel(n_channels, method='arcsinh')
            in_channels_conv = n_channels * 2
        elif scale:
            in_channels_conv = n_channels * 1
            self.preprocess = Scale(n_channels, in_channels_conv, method='arcsinh')
        else:
            self.preprocess = None
        self.inc = TripleIGConvCmplx(in_channels_conv, base_channels, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, first=True)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 4, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down3 = DownCmplx(base_channels * 4, base_channels * 8, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.down4 = DownCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, pooling=pooling)
        self.up1 = UpCmplx(base_channels * 8, base_channels * 4, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up2 = UpCmplx(base_channels * 4, base_channels * 2, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up3 = UpCmplx(base_channels * 2, base_channels, kernel_size, no_g=no_g, gp=None, relu_type=relu_type, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, no_g=no_g, gp=gp, relu_type=relu_type, mode=mode, last=True)

        gp_mult = no_g if gp is None else 1
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(nn.Sequential(
                nn.Conv2d(base_channels * 2 * gp_mult, base_channels * 2 * gp_mult, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
        self.outc = nn.Sequential(
            *linear_blocks,
            nn.Conv2d(base_channels * 2 * gp_mult, n_classes, kernel_size=1)
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
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        if self.p > 0:
            x = x[..., self.p:-self.p, self.p:-self.p]
        mask = self.outc(x)
        return mask


class UNetIGCN(Model):
    def __init__(self, n_classes, n_channels=1, base_channels=16, no_g=4,
                 kernel_size=3, mode='nearest', **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.base_channels = base_channels

        self.inc = TripleIGConv(n_channels, base_channels, kernel_size, no_g=no_g)
        self.down1 = Down(base_channels, base_channels * 2, kernel_size, no_g=no_g)
        self.down2 = Down(base_channels * 2, base_channels * 3, kernel_size, no_g=no_g)
        self.down3 = Down(base_channels * 3, base_channels * 4, kernel_size, no_g=no_g)
        self.down4 = Down(base_channels * 4, base_channels * 4, kernel_size, no_g=no_g)
        self.up1 = Up(base_channels * 4, base_channels * 3, kernel_size, no_g=no_g, mode=mode)
        self.up2 = Up(base_channels * 3, base_channels * 2, kernel_size, no_g=no_g, mode=mode)
        self.up3 = Up(base_channels * 2, base_channels, kernel_size, no_g=no_g, mode=mode)
        self.up4 = Up(base_channels, base_channels, kernel_size, no_g=no_g, mode=mode, last=True)
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        mask = self.outc(x)
        return mask


class RCF(Model):
    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16,
                 kernel_size=3, pooling='max',
                 mode='bilinear', gp='max',
                 relu_type='mod', pad_to_remove=64, **kwargs):
        super().__init__(**kwargs)
        self.p = pad_to_remove // 2
        self.to_group = IGConvCmplx(n_channels, base_channels, kernel_size, no_g=no_g, padding=1)

        self.layers = nn.ModuleList([
            RCFBlock(inc, outc, kernel_size, no_g=no_g, layers=l, gp=gp, relu_type=relu_type)
            for inc, outc, l in (
                (base_channels, base_channels, 2),
                (base_channels, base_channels * 2, 2),
                (base_channels * 2, base_channels * 2 ** 2, 3),
                (base_channels * 2 ** 2, base_channels * 2 ** 3, 3),
                (base_channels * 2 ** 3, base_channels * 2 ** 3, 3),
            )
        ])

        if pooling == 'avg':
            self.pool = AvgPoolCmplx(2)
        elif pooling == 'mag':
            self.pool = MaxMagPoolCmplx(2)
        else:
            self.pool = MaxPoolCmplx(2)

        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=2 ** i, mode=mode)
            for i in range(1, 5)
        ])

        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        x = new_cmplx(x)

        x = self.to_group(x)

        x, side = self.layers[0](x)
        if self.p > 0:
            side = side[..., self.p:-self.p, self.p:-self.p]

        sides = [side]
        for layer, up in zip(self.layers, self.ups):
            x = self.pool(x)
            x, side = layer(x)
            side = up(side)
            if self.p > 0:
                side = side[..., self.p:-self.p, self.p:-self.p]
            sides.append(side)

        x = torch.cat(sides, dim=1)

        x = self.fuse(x)

        if self.training:
            return (*sides, x)
        else:
            return x
