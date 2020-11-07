import torch, torch.nn as nn
from quicktorch.models import Model
from igcn.seg.cmplxigcn_unet_parts import DownCmplx, DownCmplxAngle, UpCmplx, TripleIGConvCmplx
from igcn.cmplx import new_cmplx, concatenate

class IGCNCovar(Model):
    __doc__ = '\n    UNet style segmentation network with orientation regression.\n    '

    def __init__(self, n_classes, n_channels=1, no_g=8, base_channels=16, kernel_size=3, nfc=1, dropout=0.0, pooling='max', mode='nearest', gp='max', angle_method=1, **kwargs):
        (super().__init__)(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        self.kernel_size = kernel_size
        self.angle_method = angle_method
        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size, no_g=no_g, gp=gp, first=True, include_gparams=True)
        self.down1 = DownCmplxAngle(base_channels, (base_channels * 2), kernel_size, no_g=no_g, gp=gp, pooling=pooling)
        self.down2 = DownCmplxAngle((base_channels * 2), (base_channels * 4), kernel_size, no_g=no_g, gp=gp, pooling=pooling)
        self.down3 = DownCmplxAngle((base_channels * 4), (base_channels * 8), kernel_size, no_g=no_g, gp=gp, pooling=pooling)
        self.down4 = DownCmplxAngle((base_channels * 8), (base_channels * 8), kernel_size, no_g=no_g, gp=gp, pooling=pooling)
        self.up1 = UpCmplx((base_channels * 8), (base_channels * 4), kernel_size, no_g=no_g, gp=gp, mode=mode)
        self.up2 = UpCmplx((base_channels * 4), (base_channels * 2), kernel_size, no_g=no_g, gp=gp, mode=mode)
        self.up3 = UpCmplx((base_channels * 2), base_channels, kernel_size, no_g=no_g, gp=gp, mode=mode)
        self.up4 = UpCmplx(base_channels, base_channels, kernel_size, no_g=no_g, mode=mode, last=True)
        linear_blocks = []
        for _ in range(nfc):
            linear_blocks.append(nn.Sequential(nn.Conv2d((base_channels * 2), (base_channels * 2), kernel_size=1), nn.ReLU(inplace=True), nn.Dropout(p=dropout)))

        self.outc = (nn.Sequential)(*linear_blocks, *(
         nn.Conv2d((base_channels * 2), n_classes, kernel_size=1),))
        self.angle_reg = nn.Sequential(nn.Linear(90, 32), nn.ReLU(True), nn.Linear(32, 1))
        if self.angle_method == 1:
            self.estimate_angle = self.reg_cat_feat_angle
        if self.angle_method == 2:
            self.angle_reg2 = nn.Linear(6, 1)
            self.estimate_angle = self.reg_feat_angle
        if self.angle_method == 3:
            self.estimate_angle = self.reg_feat
        if self.angle_method == 4:
            self.estimate_angle = self.reg_angle

    def reg_cat_feat_angle(self, feature_map, t1, t2, t3, t4, t5):
        print(feature_map.size())
        print(t1.size(), t2.size(), t3.size(), t4.size(), t5.size())
        features = torch.cat([feature_map, t1, t2, t3, t4, t5])
        angle = self.angle_reg(features)
        return angle

    def reg_feat_angle(self, feature_map, t1, t2, t3, t4, t5):
        fmap_angle = self.angle_reg(feature_map)
        angle = self.angle_reg2(torch.cat([[fmap_angle, t1, t2, t3, t4, t5]]))
        return angle

    def reg_feat(self, feature_map, t1, t2, t3, t4, t5):
        fmap_angle = self.angle_reg(feature_map)
        return fmap_angle

    def reg_angle(self, feature_map, t1, t2, t3, t4, t5):
        angle = torch.mean((t1, t2, t3, t4, t5))
        return angle

    def forward(self, x):
        print('\nINIT\n')
        x = new_cmplx(x)
        print(f"x.size()={x.size()}")
        print('\n DOWN 1\n')
        x1, t1 = self.inc(x)
        print(f"x1.size()={x1.size()}")
        print('\n DOWN 2\n')
        x2, t2 = self.down1(x1)
        print(f"x2.size()={x2.size()}")
        print('\n DOWN 3\n')
        x3, t3 = self.down2(x2)
        print(f"x3.size()={x3.size()}")
        print('\n DOWN 4\n')
        x4, t4 = self.down3(x3)
        print(f"x4.size()={x4.size()}")
        print('\n DOWN 5\n')
        x5, t5 = self.down4(x4)
        print(f"x5.size()={x5.size()}")
        print('\n UP 1\n')
        x = self.up1(x5, x4)
        print('\n UP 2\n')
        x = self.up2(x, x3)
        print('\n UP 3\n')
        x = self.up3(x, x2)
        print('\n UP 4\n')
        x = self.up4(x, x1)
        print('\n UP 5\n')
        x = torch.cat([x[0], x[1]], dim=1)
        encoding = torch.flatten(concatenate(x5))
        angle = self.estimate_angle(encoding, t1, t2, t3, t4, t5)
        angle = angle * 180
        mask = self.outc(x)
        return (mask, angle)


def main():
    pass


if __name__ == '__main__':
    main()