import sys
import torch
import torch.nn as nn
from quicktorch.utils import train
from quicktorch.models import Model
from quicktorch.data import mnist
from igcn import IGabor, IGBranched, MaxGabor, IGConv, IGParallel
import torch.optim as optim


batch_size = 2096


class Sequential(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=2),
            IGabor(no_g=4, layer=True),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7),
            IGabor(no_g=4, layer=True),
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
            IGBranched(16, 136, 7, no_g=8, padding=3),
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


class Modulated(Model):
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
            nn.Linear(576, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class Parallel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGParallel(1, 16, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGParallel(16, 32, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(288, 10)
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


class ModulatedMax(Model):
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


class ParallelMax(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGParallel(1, 16, 7, no_g=8, padding=3, max_gabor=True),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGParallel(16, 32, 7, no_g=8, padding=3, max_gabor=True),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(288, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Modulated()
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1]

    if model_type is not None:
        if model_type.lower() == 'sequential':
            net = Sequential()
        if model_type.lower() == 'branched':
            net = Branched()
        if model_type.lower() == 'modulated':
            net = Modulated()
        if model_type.lower() == 'parallel':
            net = Parallel()
        if model_type.lower() == 'branchedmax':
            net = BranchedMax()
        if model_type.lower() == 'modulatedmax':
            net = ModulatedMax()
        if model_type.lower() == 'parallelmax':
            net = ParallelMax()

    net = net.to(device)
    epochs = 5
    data, test_data, _ = mnist(batch_size=500, rotate=True)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train(net, data, opt=optimizer, device=device, epochs=epochs)


if __name__ == '__main__':
    main()
