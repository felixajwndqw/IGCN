import torch
import torch.nn.functional as F
import torch.nn as nn
from quicktorch.models import Model
from igcn.seg.attention.attention import (
    semanticModule,
    MultiConv,
    AttentionLayer,
    PAM_Module,
    CAM_Module,
    GAM_Module,
)

from igcn.seg.cmplxigcn_unet_parts import DownCmplx, TripleIGConvCmplx, UpSimpleCmplx
from igcn.seg.scale import Scale
from igcn.cmplx_modules import ReLUCmplx, IGConvGroupCmplx, Project, GaborPool
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx import new_cmplx, upsample_cmplx
from igcn.utils import _compress_shape


class ReshapeGabor(nn.Module):
    def __init__(self, gabor_dim=2):
        super().__init__()
        self.gabor_dim = gabor_dim

    def forward(self, x):
        x, _ = _compress_shape(x)
        return x


class DAFStackSmall(Model):
    def __init__(self, n_channels=1, base_channels=64, no_g=1, n_classes=1,
                 pooling='max', gp='avg', attention_gp='avg', scale=False, **kwargs):
        super().__init__(**kwargs)

        if scale:
            self.preprocess = Scale(n_channels, method='arcsinh')
        else:
            self.preprocess = None

        self.inc = TripleIGConvCmplx(n_channels, base_channels, kernel_size=3, no_g=no_g, gp=None, first=True)
        self.down1 = DownCmplx(base_channels, base_channels * 2, kernel_size=3, no_g=no_g, gp=None, pooling=pooling)
        self.down2 = DownCmplx(base_channels * 2, base_channels * 2 ** 2, kernel_size=3, no_g=no_g, gp=None, pooling=pooling)
        self.down3 = DownCmplx(base_channels * 2 ** 2, base_channels * 2 ** 3, kernel_size=3, no_g=no_g, gp=None, pooling=pooling)

        self.conv1_2 = nn.Sequential(
            IGConvGroupCmplx(base_channels * 2 ** 3, base_channels, kernel_size=1, no_g=no_g),
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            IGConvGroupCmplx(base_channels * 2 ** 2, base_channels, kernel_size=1, no_g=no_g),
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True)
        )

        self.up2 = UpSimpleCmplx(base_channels, base_channels, kernel_size=3, no_g=no_g, gp=None)
        self.up1 = UpSimpleCmplx(base_channels, base_channels, kernel_size=3, no_g=no_g, gp=None)

        self.conv8_3 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_4 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_13 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_14 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)

        self.semanticModule_1_1 = semanticModule(base_channels * 2, no_g)

        self.conv_sem_1_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_1_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        #Dual Attention mechanism
        self.pam_attention_1_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_1_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_2 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.semanticModule_2_1 = semanticModule(base_channels * 2, no_g)

        self.conv_sem_2_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_2_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        self.pam_attention_2_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_2_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_2 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.fuse1 = MultiConv(base_channels * 2, base_channels, no_g, False)

        self.refine2 = MultiConv(base_channels * 2, base_channels, no_g, False)
        self.refine1 = MultiConv(base_channels * 2, base_channels, no_g, False)

        self.predict2 = nn.Sequential(
            # GaborPool(gp),
            ReshapeGabor(),
            Project('mag'),
            nn.Conv2d(base_channels * no_g, n_classes, kernel_size=1)
        )
        self.predict1 = nn.Sequential(
            # GaborPool(gp),
            ReshapeGabor(),
            Project('mag'),
            nn.Conv2d(base_channels * no_g, n_classes, kernel_size=1)
        )

        self.predict2_2 = nn.Sequential(
            # GaborPool(gp),
            ReshapeGabor(),
            Project('mag'),
            nn.Conv2d(base_channels * no_g, n_classes, kernel_size=1)
        )
        self.predict1_2 = nn.Sequential(
            # GaborPool(gp),
            ReshapeGabor(),
            Project('mag'),
            nn.Conv2d(base_channels * no_g, n_classes, kernel_size=1)
        )

    def forward(self, down1):
        if self.preprocess is not None:
            down1 = self.preprocess(down1)
        output_size = down1.size()
        down1 = new_cmplx(down1)
        down1 = self.inc(down1)
        down1 = self.down1(down1)
        down1 = self.down2(down1)
        down2 = self.down3(down1)

        down2 = upsample_cmplx(self.conv1_2(down2), size=down1.size()[4:], mode='bilinear')
        down1 = self.conv1_1(down1)

        fuse1 = self.fuse1(torch.cat((down2, down1), 2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), dim=2))
        attention1_2 = self.conv8_3(
            (
                self.pam_attention_1_2(torch.cat((down2, fuse1), dim=2)) +
                self.cam_attention_1_2(torch.cat((down2, fuse1), dim=2)) +
                self.gam_attention_1_2(torch.cat((down2, fuse1), dim=2))
            ) *
            self.conv_sem_1_3(semanticModule_1_3)
        )

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), dim=2))
        attention1_1 = self.conv8_4(
            (
                self.pam_attention_1_1(torch.cat((down1, fuse1), dim=2)) +
                self.cam_attention_1_1(torch.cat((down1, fuse1), dim=2)) +
                self.gam_attention_1_1(torch.cat((down1, fuse1), dim=2))
            ) *
            self.conv_sem_1_4(semanticModule_1_4)
        )

        ##new design with stacked attention
        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), dim=2))
        refine2 = self.conv8_13(
            (
                self.pam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), dim=2)) +
                self.cam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), dim=2)) +
                self.gam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), dim=2))
            ) *
            self.conv_sem_2_3(semanticModule_2_3)
        )

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), dim=2))
        refine1 = self.conv8_14(
            (
                self.pam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), dim=2)) +
                self.cam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), dim=2)) +
                self.gam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), dim=2))
            ) *
            self.conv_sem_2_4(semanticModule_2_4)
        )

        predict2 = self.up1(down2)
        predict2 = self.predict2(predict2)
        predict2 = F.interpolate(predict2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1 = self.up1(down2)
        predict1 = self.predict1(predict1)
        predict1 = F.interpolate(predict1, size=output_size[2:], mode='bilinear', align_corners=True)

        predict2_2 = self.up2(refine2)
        predict2_2 = self.predict2_2(predict2_2)
        predict2_2 = F.interpolate(predict2_2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1_2 = self.up2(refine1)
        predict1_2 = self.predict1_2(predict1_2)
        predict1_2 = F.interpolate(predict1_2, size=output_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            return (
                (
                    semVector_1_3,
                    semVector_2_3,
                ), (
                    semVector_1_4,
                    semVector_2_4,
                ), (
                   torch.cat((down1, fuse1), 2),
                   torch.cat((down2, fuse1), 2),
                   torch.cat((down1, attention1_1 * fuse1), 2),
                   torch.cat((down2, attention1_2 * fuse1), 2),
                ), (
                   semanticModule_1_4,
                   semanticModule_1_3,
                   semanticModule_2_4,
                   semanticModule_2_3,
                ), (
                   predict1,
                   predict2,
                   predict1_2,
                   predict2_2
                )
            )
        else:
            return ((predict1_2 + predict2_2) / 2)
