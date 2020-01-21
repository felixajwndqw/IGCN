from igcn import gabor
import torch.nn as nn
import torch
from quicktorch.utils import train
from quicktorch.models import Model
from quicktorch.data import mnist
from igcn import GaborFunction
import matplotlib.pyplot as plt


count = 0
batch_size = 2096


class Simple(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 10)
        )
        self.classifier[0].name = "simple"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class Gabor(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 10)
        )
        self.classifier[0].name = "gabor"

    def forward(self, x):
        x = self.relu(GaborFunction.apply(x))
        x = self.relu(GaborFunction.apply(x))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class SimpleWithGabor(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 10)
        )
        self.classifier[0].name = "swg"

    def forward(self, x):
        x = self.relu(GaborFunction.apply(self.conv1(x)))
        x = self.relu(GaborFunction.apply(self.conv2(x)))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def printgradnorm(self, grad_input, grad_output):
    print('grad_input norm:', grad_input[0].norm())
    print('grad_output norm:', grad_output[0].norm())


def plot(self, input, output):
    global count
    with torch.no_grad():
        print("Count:", count)
        print("Input:", input[0].size())
        plt.clf()
        x = input[0].clone().detach().cpu().view(-1, 32, 28, 28)[0, 0]
        plt.imshow(x)
        fname = "figs/filters/{}_c{}".format(self.name, count)
        count = count + 1
        plt.savefig(fname)


def main():
    global count
    models = [Simple(name='simple').cuda(),
              Gabor(name='gabor').cuda(),
              SimpleWithGabor(name='swg').cuda()]
    train_data, test_data, _ = mnist(rotate=True, batch_size=batch_size)

    results = []
    for m in models:
        m.classifier[0].register_forward_hook(plot)
        # m.features[-2].register_backward_hook(printgradnorm)
        o = train(m, train_data, epochs=5, device=0)
        results.append(o)
        count = 0

    print([oi for oi in o])

if __name__ == '__main__':
    main()
