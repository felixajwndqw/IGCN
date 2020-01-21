import torch.nn as nn
import torch
from quicktorch.utils import train
from quicktorch.models import Model
from quicktorch.data import mnist
from torch.nn.modules.conv import _ConvNd
import torch.nn.functional as F
from igcn.vis import FilterPlot
import torch.optim as optim


batch_size = 2096


class IGConv(_ConvNd):
    def __init__(self, input_features, output_features, kernel_size,
                 stride=1, padding=0, dilation=1, bias=None):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super().__init__(
            input_features, output_features, kernel_size,
            stride, padding, dilation, False, (0, 0), 1, bias, 'zeros'
        )
        self.plot = None

    def forward(self, x):
        out = F.conv2d(x, self.weight, None, self.stride,
                       self.padding, self.dilation)
        if self.plot is not None:
            self.plot.update(self.weight[:, 0].clone().detach().cpu().numpy(),
                             torch.zeros(4, *self.kernel_size),
                             torch.zeros(2, 1))
        return out


class Simple(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.pooling = nn.MaxPool2d(kernel_size=3)
        self.conv1 = IGConv(1, 16, 9, stride=2)
        # self.conv1 = nn.Conv2d(1, 16, 9, stride=2)
        self.conv2 = IGConv(16, 32, 9, stride=2)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(32, 10)
        )
        self.classifier[0].name = "simple"

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def visualise_layer(layer, print_pixel=False):
    layer.plot = FilterPlot(1, layer.kernel_size[0],
                            layer.out_channels, ion=True)
    if print_pixel:
        layer.register_forward_hook(print_first_pixel)


def print_first_pixel(self, input, output):
    print(self.weight[0, 0, 0, 0:8].tolist())


def main():
    device = 0
    simple = Simple().to(device)

    #  Register vis.FilterPlot plotting function
    visualise_layer(simple.conv1)

    epochs = 5
    data, test_data, _ = mnist(batch_size=500)
    optimizer = optim.Adam(simple.parameters(), lr=1e-3)
    train(simple, data, opt=optimizer, device=device, epochs=epochs)


if __name__ == '__main__':
    main()
