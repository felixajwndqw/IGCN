import torch, torch.nn as nn
from quicktorch.models import Model
from igcn.seg.cmplxigcn_unet_parts import DownCmplx, DownCmplxAngle, UpCmplx, TripleIGConvCmplx
from igcn.cmplx import new_cmplx, concatenate
from igcn.cmplx_modules import IGConvGroupCmplx, ReLUCmplx, MaxPoolCmplx

from igcn.cmplx import cmplx
from igcn.cmplx_modules import ConvCmplx
from igcn.utils import _compress_shape, _recover_shape


class IGCNCovar(Model):
    __doc__ = '\n    UNet style segmentation network with orientation regression.\n    '

    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16, kernel_size=3, nfc=1, dropout=0.0, pooling='max', mode='nearest', gp='max', angle_method=1, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.angle_method = angle_method
        # self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g, gp=gp, first=True, include_gparams=True)
        # self.down1 = DownCmplxAngle(base_channels, base_channels * 2, kernel_size, no_g, gp=gp, pooling=pooling)
        # self.down2 = DownCmplxAngle(base_channels * 2, base_channels * 4, kernel_size, no_g, gp=gp, pooling=pooling)
        # self.down3 = DownCmplxAngle(base_channels * 4, base_channels * 8, kernel_size, no_g, gp=gp, pooling=pooling)
        # self.down4 = DownCmplxAngle(base_channels * 8, base_channels * 8, kernel_size, no_g, gp=gp, pooling=pooling)
        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g, gp=None, first=True)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size, no_g, gp=None, pooling=pooling)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 4, kernel_size, no_g, gp=None, pooling=pooling)
        self.down3 = DownCmplx(base_channels * 4, base_channels * 8, kernel_size, no_g, gp=None, pooling=pooling)
        self.down4 = DownCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g, gp=None, pooling=pooling)
        self.up1 = UpCmplx(base_channels * 8, base_channels * 4, kernel_size, no_g, mode=mode)
        self.up2 = UpCmplx(base_channels * 4, base_channels * 2, kernel_size, no_g, mode=mode)
        self.up3 = UpCmplx(base_channels * 2, base_channels, kernel_size, no_g, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, no_g, mode=mode, gp=gp)
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(
                nn.Sequential(
                    nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout)
                )
            )

        self.outc = nn.Sequential(
            *linear_blocks,
            nn.Conv2d(base_channels * 2, n_classes, kernel_size=1),
        )
        self.loc_net = nn.Sequential(
            IGConvGroupCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g, padding=1),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(2),
            IGConvGroupCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(2),
            IGConvGroupCmplx(base_channels * 8, base_channels * 8, kernel_size, no_g, gabor_pooling=gp),
            ReLUCmplx(inplace=True),
        )
        self.angle_reg = nn.Sequential(
            nn.Linear(16 * base_channels, 1),
            nn.ReLU(inplace=True)
        )
        # if self.angle_method == 1:
        #     self.estimate_angle = self.reg_cat_feat_angle
        # if self.angle_method == 2:
        #     self.angle_reg2 = nn.Linear(6, 1)
        #     self.estimate_angle = self.reg_feat_angle
        if self.angle_method == 3:
            self.estimate_angle = self.reg_feat
        # if self.angle_method == 4:
        #     self.estimate_angle = self.reg_angle

    # def reg_cat_feat_angle(self, feature_map, thetas):
    #     # print(feature_map.size())
    #     # print([f'thetas[{i}].size()={thetas[i].size()}' for i in range(5)])
    #     features = torch.cat([feature_map, thetas])
    #     # print(features.size())
    #     angle = self.angle_reg(features)
    #     return angle

    # def reg_feat_angle(self, feature_map, thetas):
    #     thetas = torch.stack([thetas_i.flatten(1).mean(1) for thetas_i in thetas], dim=1)
    #     # print(f'thetas.size()={thetas.size()}, feature_map.size()={feature_map.size()}')
    #     print("BEFORE LOCNET")
    #     print(f'feature_map.min()={feature_map.min()}, '
    #           f'feature_map.max()={feature_map.max()}, '
    #           f'feature_map.mean()={feature_map.mean()}')
    #     feature_map = self.loc_net(feature_map)
    #     print("AFTER LOCNET")
    #     print(f'feature_map.min()={feature_map.min()}'
    #           f'feature_map.max()={feature_map.max()}'
    #           f'feature_map.mean()={feature_map.mean()}')
    #     # print(f'feature_map.size()={feature_map.size()}')
    #     feature_map = concatenate(feature_map.flatten(2))
    #     # print(f'feature_map.size()={feature_map.size()}')
    #     feature_map_angle = self.angle_reg(feature_map)
    #     print(f'feature_map_angle={feature_map_angle}, thetas={thetas}')
    #     # print(f'feature_map_angle.size()={feature_map_angle.size()}')
    #     angle = self.angle_reg2(torch.cat([feature_map_angle, thetas], dim=1))
    #     return angle

    # def reg_feat(self, feature_map, thetas):
    #     # print("BEFORE LOCNET")
    #     # print(f'feature_map.min()={feature_map.min()}, '
    #     #       f'feature_map.max()={feature_map.max()}, '
    #     #       f'feature_map.mean()={feature_map.mean()}')
    #     feature_map = self.loc_net(feature_map)
    #     # print("AFTER LOCNET")
    #     # print(f'feature_map.min()={feature_map.min()}, '
    #     #       f'feature_map.max()={feature_map.max()}, '
    #     #       f'feature_map.mean()={feature_map.mean()}')
    #     feature_map = concatenate(feature_map.flatten(2))
    #     feature_map_angle = self.angle_reg(feature_map)
    #     return feature_map_angle

    def reg_feat(self, feature_map):
        # print("BEFORE LOCNET")
        # print(f'feature_map.min()={feature_map.min()}, '
        #       f'feature_map.max()={feature_map.max()}, '
        #       f'feature_map.mean()={feature_map.mean()}')
        feature_map = self.loc_net(feature_map)
        feature_map = concatenate(feature_map.flatten(2))
        feature_map_angle = self.angle_reg(feature_map)
        # print("AFTER LOCNET")
        # print(f'feature_map.min()={feature_map.min()}, '
        #       f'feature_map.max()={feature_map.max()}, '
        #       f'feature_map.mean()={feature_map.mean()}')
        return feature_map_angle

    # def reg_angle(self, feature_map, thetas):
    #     thetas = torch.stack(thetas)
    #     angle = torch.mean(thetas)
    #     return angle

    def forward(self, x):
        # print('\nINIT\n')
        x = new_cmplx(x)
        # print(f"x.size()={x.size()}")
        # print('\n DOWN 1\n')
        x1 = self.inc(x)
        # print(f"x1.size()={x1.size()}")
        # print('\n DOWN 2\n')
        x2 = self.down1(x1)
        # print(f"x2.size()={x2.size()}")
        # print('\n DOWN 3\n')
        x3 = self.down2(x2)
        # print(f"x3.size()={x3.size()}")
        # print('\n DOWN 4\n')
        x4 = self.down3(x3)
        # print(f"x4.size()={x4.size()}")
        # print('\n DOWN 5\n')
        x5 = self.down4(x4)
        # print(f"x5.size()={x5.size()}")
        # print('\n UP 1\n')
        x = self.up1(x5, x4)
        # print('\n UP 2\n')
        x = self.up2(x, x3)
        # print('\n UP 3\n')
        x = self.up3(x, x2)
        # print('\n UP 4\n')
        x = self.up4(x, x1)
        # print('\n FINAL\n')
        # print(f"x.size()={x.size()}")
        x = concatenate(x)
        angle = self.estimate_angle(x5)
        angle = angle * 180
        mask = self.outc(x)
        return mask, angle

    def forward2(self, x):
        # print('\nINIT\n')
        x = new_cmplx(x)
        # print(f"x.size()={x.size()}")
        # print('\n DOWN 1\n')
        x1, t1 = self.inc(x)
        # print(f"x1.size()={x1.size()}")
        # print('\n DOWN 2\n')
        x2, t2 = self.down1(x1)
        # print(f"x2.size()={x2.size()}")
        # print('\n DOWN 3\n')
        x3, t3 = self.down2(x2)
        # print(f"x3.size()={x3.size()}")
        # print('\n DOWN 4\n')
        x4, t4 = self.down3(x3)
        # print(f"x4.size()={x4.size()}")
        # print('\n DOWN 5\n')
        x5, t5 = self.down4(x4)
        # print(f"x5.size()={x5.size()}")
        # print('\n UP 1\n')
        x = self.up1(x5, x4)
        # print('\n UP 2\n')
        x = self.up2(x, x3)
        # print('\n UP 3\n')
        x = self.up3(x, x2)
        # print('\n UP 4\n')
        x = self.up4(x, x1)
        # print('\n FINAL\n')
        # print(f"x.size()={x.size()}")
        x = concatenate(x)
        thetas = [t1, t2, t3, t4, t5]
        angle = self.estimate_angle(x5, thetas)
        angle = angle * 180
        # print(f"x.size()={x.size()}")
        mask = self.outc(x)
        return mask, angle


class IGCNCovarTest(Model):
    __doc__ = '\n    UNet style segmentation network with orientation regression.\n    '

    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16, kernel_size=3, nfc=1, dropout=0.0, pooling='max', mode='nearest', gp='max', angle_method=1, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.angle_method = angle_method
        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g, gp=None, first=True)
        self.down = DownCmplx(base_channels, base_channels, kernel_size, no_g, gp=None, pooling=pooling)
        self.up = UpCmplx(base_channels, base_channels, kernel_size, no_g, mode=mode, gp=gp)
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(
                nn.Sequential(
                    nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout)
                )
            )

        self.outc = nn.Sequential(
            *linear_blocks,
            nn.Conv2d(base_channels * 2, n_classes, kernel_size=1),
        )
        self.loc_net = nn.Sequential(
            IGConvGroupCmplx(base_channels, base_channels, kernel_size, no_g, padding=1),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(4),
            IGConvGroupCmplx(base_channels, base_channels, kernel_size, no_g),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(4),
            IGConvGroupCmplx(base_channels, base_channels, kernel_size, no_g, gabor_pooling=gp),
            ReLUCmplx(inplace=True),
        )
        self.angle_reg = nn.Sequential(
            nn.Linear(100, 1),
            nn.ReLU(inplace=True)
        )

    def reg_feat(self, feature_map):
        # print("BEFORE LOCNET")
        # print(f'feature_map.min()={feature_map.min()}, '
        #       f'feature_map.max()={feature_map.max()}, '
        #       f'feature_map.mean()={feature_map.mean()}')
        feature_map = self.loc_net(feature_map)
        feature_map_cat = concatenate(feature_map.flatten(2))
        feature_map_angle = self.angle_reg(feature_map_cat)
        # print("AFTER LOCNET")
        # print(f'feature_map.min()={feature_map.min()}, '
        #       f'feature_map.max()={feature_map.max()}, '
        #       f'feature_map.mean()={feature_map.mean()}')
        return feature_map_angle

    def forward(self, x):
        # print('\nINIT\n')
        x = new_cmplx(x)
        # print(f"x.size()={x.size()}")
        # print('\n DOWN 1\n')
        x1 = self.inc(x)
        # print(f"x1.size()={x1.size()}")
        # print('\n DOWN 2\n')
        x2 = self.down(x1)
        # print('\n UP 1\n')
        x = self.up(x2, x1)
        # print('\n FINAL\n')
        # print(f"x.size()={x.size()}")
        x = concatenate(x)
        # print(f"x.size()={x.size()}")
        angle = self.reg_feat(x2)
        angle = angle * 180
        mask = self.outc(x)
        return mask, angle


class IGCNCovarTest2(Model):
    __doc__ = '\n    UNet style segmentation network with orientation regression.\n    '

    def __init__(
        self,
        n_classes,
        n_channels=1,
        base_channels=16,
        kernel_size=3,
        nfc=1,
        dropout=0.0,
        pooling='max',
        mode='nearest',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.inc = ConvCmplx(n_channels, base_channels, kernel_size, padding=1)
        self.down = DownCmplxTest(base_channels, base_channels, kernel_size, pooling=pooling)
        self.up = UpCmplxTest(base_channels, base_channels, kernel_size, mode=mode)
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(
                nn.Sequential(
                    nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout)
                )
            )

        self.outc = nn.Sequential(
            *linear_blocks,
            nn.Conv2d(base_channels * 2, n_classes, kernel_size=1),
        )
        self.loc_net = nn.Sequential(
            ConvCmplx(base_channels, base_channels, kernel_size, padding=1),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(4),
            ConvCmplx(base_channels, base_channels, kernel_size),
            ReLUCmplx(inplace=True),
            MaxPoolCmplx(4),
            ConvCmplx(base_channels, base_channels, kernel_size),
            ReLUCmplx(inplace=True),
        )
        self.angle_reg = nn.Sequential(
            nn.Linear(100, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = new_cmplx(x)
        x1 = self.inc(x)
        x2 = self.down(x1)

        x3 = self.up(x2, x1)
        x4 = concatenate(x3)
        mask = self.outc(x4)

        feature_map = self.loc_net(x2)
        feature_map_cat = concatenate(feature_map.flatten(2))
        angle = self.angle_reg(feature_map_cat)
        angle = angle * 180

        return mask, angle


class DownCmplxTest(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='avg'):
        super().__init__()
        Pool = MaxPoolCmplx(2)
        self.pool_conv = nn.Sequential(
            Pool,
            ConvCmplx(in_channels, out_channels, kernel_size, padding=1)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpCmplxTest(nn.Module):
    """Upscaling then double conv

    Args:
        in_channels:
        out_channels
        mode (str, optional): Upsampling method. If None ConvTranspose2d will
            be used. Defaults to 'nearest'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, mode='nearest'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = ConvCmplx(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x1, x2):
        x1, xs = _compress_shape(x1)

        x1 = cmplx(self.up(x1[0]), self.up(x1[1]))

        x1 = _recover_shape(x1, xs)

        return self.conv(x1 + x2)


def main():
    pass


if __name__ == '__main__':
    main()