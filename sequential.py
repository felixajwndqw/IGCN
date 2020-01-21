import torch.nn as nn
from quicktorch.utils import train
from quicktorch.models import Model
from quicktorch.data import mnist
from igcn import IGabor, IGBranched, MaxGabor, IGConv
import torch.optim as optim


batch_size = 2096


class Sequential(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=2),
            IGabor(no_g=4),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7),
            IGabor(no_g=4),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class Branched(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGBranched(1, 16, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGBranched(24, 40, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2088, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class Normal(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGConv(1, 32, 7, no_g=4, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGConv(32, 64, 7, no_g=4, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class BranchedMax(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGBranched(1, 16, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGBranched(24, 40, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2088, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class NormalMax(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGConv(1, 32, 7, no_g=4, padding=3),
            MaxGabor(no_g=4),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGConv(40, 64, 7, no_g=4, padding=3),
            MaxGabor(no_g=4),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    device = 0
    # net = Sequential().to(device)
    net = NormalMax().to(device)

    epochs = 5
    data, test_data, _ = mnist(batch_size=500)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train(net, data, opt=optimizer, device=device, epochs=epochs)


if __name__ == '__main__':
    main()
