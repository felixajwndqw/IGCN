import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_parts import Up, Down, DoubleIGConv
from quicktorch.models import Model
from quicktorch.utils import train, imshow
from data import CirrusDataset


class UNetIGCN(Model):
    def __init__(self, n_channels, n_classes, bilinear=True, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleIGConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

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


if __name__ == '__main__':
    train_cirri = DataLoader(CirrusDataset('data/cirrus_examples/train'),
                             batch_size=2, shuffle=True)
    test_cirri = DataLoader(CirrusDataset('data/cirrus_examples/test'),
                            batch_size=1, shuffle=True)
    model = UNetIGCN(n_channels=1, n_classes=1)
    optimizer = optim.Adam(model.parameters())
    train(model, train_cirri, epochs=1, opt=optimizer)
    # train(model, [train_cirri, test_cirri], epochs=1, opt=optimizer)

    # example = iter(test_cirri).next()
    # test_out = model(example[0])
    # print(test_out.size())
    # imshow(example[0].detach(), test_out.detach())
