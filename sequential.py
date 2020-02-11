import sys
import time
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
            IGabor(no_g=8, layer=True),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 7),
            IGabor(no_g=8, layer=True),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 10)
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
            IGBranched(16, 64, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1179, 10)
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
            IGConv(1, 32, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGConv(32, 64, 7, no_g=8, padding=3),
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


class Parallel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGParallel(1, 32, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
            IGParallel(32, 64, 7, no_g=8, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class ParallelMore(Model):
    def __init__(self, no_g=4, **kwargs):
        super().__init__(**kwargs)
        self.no_g = no_g
        self.features = nn.Sequential(
            IGParallel(1, 16, 7, no_g=no_g, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            IGParallel(16, 32, 7, no_g=no_g, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            IGParallel(32, 48, 7, no_g=no_g, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            IGParallel(48, 64, 7, no_g=no_g, padding=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 10)
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


def write_results(model_name, no_g, m, no_epochs,
                  mins, secs, cmplx=False):
    f = open("seq_results.txt", "a+")
    f.write("\n" + model_name +
            "," + str(no_g) +
            "," + str(cmplx) +
            ',' + "{:1.4f}".format(m['accuracy']) +
            "," + "{:1.4f}".format(m['precision']) +
            "," + "{:1.4f}".format(m['recall']) +
            "," + str(m['epoch']) +
            "," + str(no_epochs) +
            ',' + "{:3d}m{:2d}s".format(mins, secs))
    f.close()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nets = []
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1].split(',')

    if model_type is not None:
        for m in model_type:
            if m.lower() == 'sequential':
                nets.append(Sequential(name='sequential'))
            if m.lower() == 'branched':
                nets.append(Branched(name='branched'))
            if m.lower() == 'modulated':
                nets.append(Modulated(name='modulated'))
            if m.lower() == 'parallel':
                nets.append(Parallel(name='parallel'))
            if m.lower() == 'parallelmore':
                nets.append(ParallelMore(name='parallelmore'))
            if m.lower() == 'branchedmax':
                nets.append(BranchedMax())
            if m.lower() == 'modulatedmax':
                nets.append(ModulatedMax())
            if m.lower() == 'parallelmax':
                nets.append(ParallelMax())
    else:
        nets.append(Modulated())

    for net in nets:
        net.save_dir = 'models/seq_tests/'
        net = net.to(device)
        epochs = 250
        data, test_data, _ = mnist(batch_size=2048, rotate=True)

        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        start = time.time()
        m = train(net, data, opt=optimizer, device=device, epochs=epochs,
                  sch=scheduler, save_best=True)

        time_taken = time.time() - start
        mins = int(time_taken // 60)
        secs = int(time_taken % 60)
        write_results(net.name, 8, m, epochs, mins, secs)


if __name__ == '__main__':
    main()
