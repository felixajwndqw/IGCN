import torch
import torch.nn.functional as F
import torch.nn as nn
from quicktorch.models import Model
from igcn.seg.attention.attention import (
    Disassemble,
    DisassembleCmplx,
    Reassemble,
    ReassembleCmplx,
    SemanticModule,
    MultiConv,
    AttentionLayer,
    PAM_Module,
    CAM_Module,
    GAM_Module,
)

from igcn.seg.cmplx_modules import DownCmplx, TripleIGConvCmplx, UpSimpleCmplx
from igcn.cmplx_modules import MaxPoolCmplx, ReLUCmplx, IGConvGroupCmplx, Project, GaborPool, IGConvCmplx, ReshapeGabor
from igcn.cmplx_bn import BatchNormCmplx
from igcn.cmplx import new_cmplx, resample_cmplx
from igcn.utils import _compress_shape


class DAFStackSmall(Model):
    def __init__(self, n_channels=1, base_channels=64, no_g=1, n_classes=1,
                 pooling='max', gp='avg', attention_gp='avg', scale=None, **kwargs):
        super().__init__(**kwargs)

        self.preprocess = scale

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

        self.semanticModule_1_1 = SemanticModule(base_channels * 2, no_g)

        self.conv_sem_1_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_1_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        # Dual Attention mechanism
        self.pam_attention_1_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_1_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_2 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.semanticModule_2_1 = SemanticModule(base_channels * 2, no_g)

        self.conv_sem_2_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_2_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        self.pam_attention_2_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_2_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_2 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.fuse1 = MultiConv(base_channels * 2, base_channels, no_g, False)

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

        down2 = resample_cmplx(self.conv1_2(down2), size=down1.size()[4:], mode='bilinear')
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
                    semVector_1_4,
                ), (
                    semVector_2_3,
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


class DAFMS(Model):
    def __init__(self, n_channels=1, base_channels=64, no_g=1, n_classes=1,
                 pooling='max', gp='avg', attention_gp='avg', pad_to_remove=64,
                 scale=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.preprocess = scale

        self.disassemble = Disassemble()
        self.disassemble_cmplx = DisassembleCmplx()
        self.reassemble = Reassemble()
        self.reassemble_cmplx = ReassembleCmplx()
        self.p = pad_to_remove // 2

        self.down1 = nn.Sequential(
            IGConvCmplx(n_channels, base_channels, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            IGConvGroupCmplx(base_channels, base_channels, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            MaxPoolCmplx(2),
            IGConvGroupCmplx(base_channels, base_channels * 2, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * 2 * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            IGConvGroupCmplx(base_channels * 2, base_channels * 2, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * 2 * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            MaxPoolCmplx(2),
            IGConvGroupCmplx(base_channels * 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
            BatchNormCmplx(base_channels * 2 ** 2 * no_g),
            ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
            IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
            BatchNormCmplx(base_channels * 2 ** 2 * no_g),
            ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
            MaxPoolCmplx(2),
            IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * 2 ** 3 * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            IGConvGroupCmplx(base_channels * 2 ** 3, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
            BatchNormCmplx(base_channels * 2 ** 3 * no_g),
            ReLUCmplx(channels=base_channels, inplace=True),
            MaxPoolCmplx(2),
        )
        # self.down2 = nn.Sequential(
        #     IGConvCmplx(n_channels, base_channels, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels, base_channels, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels, base_channels * 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 3, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        # )
        # self.down3 = nn.Sequential(
        #     IGConvCmplx(n_channels, base_channels, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels, base_channels, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels, base_channels * 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1), # ** 3
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels * 2 ** 2, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 3, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        # )
        # self.align2 = nn.Sequential(
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        # )
        # self.align3 = nn.Sequential(
        #     IGConvGroupCmplx(base_channels * 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 2, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 2 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        #     IGConvGroupCmplx(base_channels * 2 ** 2, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     IGConvGroupCmplx(base_channels * 2 ** 3, base_channels * 2 ** 3, 3, no_g=no_g, padding=1),
        #     BatchNormCmplx(base_channels * 2 ** 3 * no_g),
        #     ReLUCmplx(channels=base_channels, inplace=True),
        #     MaxPoolCmplx(2),
        # )
        self.conv1_3 = nn.Sequential(
            IGConvGroupCmplx(base_channels * 2 ** 3, base_channels, kernel_size=1, no_g=no_g), # ** 3
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            IGConvGroupCmplx(base_channels * 2 ** 3, base_channels, kernel_size=1, no_g=no_g), # ** 2
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            IGConvGroupCmplx(base_channels * 2 ** 3, base_channels, kernel_size=1, no_g=no_g),
            BatchNormCmplx(base_channels * no_g),
            ReLUCmplx(channels=base_channels, inplace=True)
        )

        self.up2 = UpSimpleCmplx(base_channels, base_channels, kernel_size=3, no_g=no_g, gp=None)
        self.up1 = UpSimpleCmplx(base_channels, base_channels, kernel_size=3, no_g=no_g, gp=None)

        self.conv8_2 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_3 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_4 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_12 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_13 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)
        self.conv8_14 = IGConvGroupCmplx(base_channels, base_channels, kernel_size=1, no_g=no_g)

        self.semanticModule_1_1 = SemanticModule(base_channels * 2, no_g)

        self.conv_sem_1_2 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_1_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_1_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        #Dual Attention mechanism
        self.pam_attention_1_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_1_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_2 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_1_3 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_1_3 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_1_3 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.semanticModule_2_1 = SemanticModule(base_channels * 2, no_g)

        self.conv_sem_2_2 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_2_3 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)
        self.conv_sem_2_4 = IGConvGroupCmplx(base_channels * 2, base_channels, kernel_size=3, padding=1, no_g=no_g)

        self.pam_attention_2_1 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_1 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_1 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_2_2 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_2 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_2 = AttentionLayer(base_channels, no_g, GAM_Module)
        self.pam_attention_2_3 = AttentionLayer(base_channels, no_g, PAM_Module, gp=attention_gp)
        self.cam_attention_2_3 = AttentionLayer(base_channels, no_g, CAM_Module)
        self.gam_attention_2_3 = AttentionLayer(base_channels, no_g, GAM_Module)

        self.fuse1 = MultiConv(3 * base_channels, base_channels, no_g, False)

        self.predict3 = nn.Sequential(
            # GaborPool(gp),
            ReshapeGabor(),
            Project('mag'),
            nn.Conv2d(base_channels * no_g, n_classes, kernel_size=1)
        )
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

        self.predict3_2 = nn.Sequential(
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

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        output_size = x.size()
        # Create downscaled copies
        down3 = x
        down2 = F.interpolate(
            x,
            scale_factor=1/2
        )
        down1 = F.interpolate(
            x,
            scale_factor=1/4
        )

        # Generate features
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')
        down1 = self.down1(new_cmplx(down1))
        down2 = self.down1(new_cmplx(down2))
        down3 = self.down1(new_cmplx(down3))

        # down2 = self.align2(down2)
        # down3 = self.align3(down3)
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        # down2 = self.reassemble_cmplx(down2)
        # down3 = self.reassemble_cmplx(self.reassemble_cmplx(down3))
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        down1 = self.conv1_1(down1)
        down2 = self.conv1_2(down2)
        down3 = self.conv1_3(down3)
        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')

        # Align scales
        fuse1 = self.fuse1(torch.cat((
            down3,
            resample_cmplx(down2, down3.size()[-2:]),
            resample_cmplx(down1, down3.size()[-2:])
        ), 2))
        # print(f'{fuse1.size()=}')

        fuse1_3 = self.disassemble_cmplx(self.disassemble_cmplx(fuse1))
        fuse1_2 = self.disassemble_cmplx(resample_cmplx(fuse1, size=down2.size()[-2:])) 
        fuse1_1 = resample_cmplx(fuse1, size=down1.size()[-2:])
        # print(f'{fuse1_1.size()=}, {fuse1_2.size()=}, {fuse1_3.size()=}')

        fused1_3 = torch.cat((
            self.disassemble_cmplx(self.disassemble_cmplx(down3)),
            self.disassemble_cmplx(self.disassemble_cmplx(fuse1))
        ), dim=2)
        # print(f'First attention: {fused1_3.size()=}')
        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(fused1_3)
        attention1_3 = self.conv8_2(
            (
                self.pam_attention_1_3(fused1_3) +
                self.cam_attention_1_3(fused1_3) +
                self.gam_attention_1_3(fused1_3)
            ) *
            self.conv_sem_1_2(semanticModule_1_2)
        )
        # print(f'{attention1_3.size()=}')

        fused2_3 = torch.cat((
            self.disassemble_cmplx(self.disassemble_cmplx(down3)),
            attention1_3 * self.disassemble_cmplx(self.disassemble_cmplx(fuse1))
        ), dim=2)
        # print(f'First refine: {fused2_3.size()=}')
        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(fused2_3)
        refine3 = self.conv8_12(
            (
                self.pam_attention_2_3(fused2_3) +
                self.cam_attention_2_3(fused2_3) +
                self.gam_attention_2_3(fused2_3)
            ) *
            self.conv_sem_2_2(semanticModule_2_2)
        )
        # print(f'{refine3.size()=}')

        fused1_2 = torch.cat((
            self.disassemble_cmplx(down2),
            self.disassemble_cmplx(resample_cmplx(fuse1, size=down2.size()[-2:])) 
        ), dim=2)
        # print(f'Second attention: {fused1_2.size()=}')
        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(fused1_2)
        attention1_2 = self.conv8_3(
            (
                self.pam_attention_1_2(fused1_2) +
                self.cam_attention_1_2(fused1_2) +
                self.gam_attention_1_2(fused1_2)
            ) *
            self.conv_sem_1_3(semanticModule_1_3)
        )
        # print(f'{attention1_2.size()=}')

        fused2_2 = torch.cat((
            self.disassemble_cmplx(down2),
            attention1_2 * self.disassemble_cmplx(resample_cmplx(fuse1, size=down2.size()[-2:])) 
        ), dim=2)
        # print(f'Second refine: {fused2_2.size()=}')
        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(fused2_2)
        refine2 = self.conv8_13(
            (
                self.pam_attention_2_2(fused2_2) +
                self.cam_attention_2_2(fused2_2) +
                self.gam_attention_2_2(fused2_2)
            ) *
            self.conv_sem_2_3(semanticModule_2_3)
        )
        # print(f'{refine2.size()=}')

        fused1_1 = torch.cat((
            down1,
            resample_cmplx(fuse1, size=down1.size()[-2:])
        ), dim=2)
        # print(f'Third attention: {fused1_1.size()=}')
        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(fused1_1)
        attention1_1 = self.conv8_4(
            (
                self.pam_attention_1_1(fused1_1) +
                self.cam_attention_1_1(fused1_1) +
                self.gam_attention_1_1(fused1_1)
            ) *
            self.conv_sem_1_4(semanticModule_1_4)
        )
        # print(f'{attention1_1.size()=}')

        fused2_1 = torch.cat((
            down1,
            attention1_1 * resample_cmplx(fuse1, size=down1.size()[-2:])
        ), dim=2)
        # print(f'Third refine: {fused2_1.size()=}')
        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(fused2_1)
        refine1 = self.conv8_14(
            (
                self.pam_attention_2_1(fused2_1) +
                self.cam_attention_2_1(fused2_1) +
                self.gam_attention_2_1(fused2_1)
            ) *
            self.conv_sem_2_4(semanticModule_2_4)
        )
        # print(f'{refine1.size()=}')

        # print(f'{down1.size()=}, {down2.size()=}, {down3.size()=}')
        predict3 = self.up1(down3)
        predict2 = self.up1(down2)
        predict1 = self.up1(down1)

        if self.p > 0:
            predict3 = predict3[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict2 = predict2[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict1 = predict1[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
        # print(f'{predict1.size()=}, {predict2.size()=}, {predict3.size()=}')

        predict3 = self.predict3(predict3)
        predict2 = self.predict2(predict2)
        predict1 = self.predict1(predict1)
        # print(f'{predict1.size()=}, {predict2.size()=}, {predict3.size()=}')
        predict3 = F.interpolate(predict3, size=output_size[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=output_size[2:], mode='bilinear', align_corners=True)

        if self.p > 0:
            predict3 = predict3[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict2 = predict2[..., self.p//4:-self.p//4, self.p//4:-self.p//4]
            predict1 = predict1[..., self.p//4:-self.p//4, self.p//4:-self.p//4]

        refine3 = self.reassemble_cmplx(self.reassemble_cmplx(refine3))
        refine2 = self.reassemble_cmplx(refine2)
        # print(f'{refine1.size()=}, {refine2.size()=}, {refine3.size()=}')

        predict3_2 = self.up2(refine3)
        predict2_2 = self.up2(refine2)
        predict1_2 = self.up2(refine1)
        # print(f'{predict1_2.size()=}, {predict2_2.size()=}, {predict3_2.size()=}')

        predict3_2 = self.predict3_2(predict3_2)
        predict2_2 = self.predict2_2(predict2_2)
        predict1_2 = self.predict1_2(predict1_2)
        # print(f'{predict1_2.size()=}, {predict2_2.size()=}, {predict3_2.size()=}')

        predict3_2 = F.interpolate(predict3_2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict2_2 = F.interpolate(predict2_2, size=output_size[2:], mode='bilinear', align_corners=True)
        predict1_2 = F.interpolate(predict1_2, size=output_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            return (
                (
                    semVector_1_2,
                    semVector_1_3,
                    semVector_1_4,
                ), (
                    semVector_2_2,
                    semVector_2_3,
                    semVector_2_4,
                ), (
                   fused1_1,
                   fused1_2,
                   fused1_3,
                   fused2_1,
                   fused2_2,
                   fused2_3,
                ), (
                   semanticModule_1_4,
                   semanticModule_1_3,
                   semanticModule_1_2,
                   semanticModule_2_4,
                   semanticModule_2_3,
                   semanticModule_2_2,
                ), (
                   predict1,
                   predict2,
                   predict3,
                   predict1_2,
                   predict2_2,
                   predict3_2,
                )
            )
        else:
            return ((predict1_2 + predict2_2 + predict3_2) / 3)
