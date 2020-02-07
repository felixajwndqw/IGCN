import sys
import torch
import torch.nn as nn
import torch.optim as optim
from igcn.cmplx_modules import (
    IGConvCmplx,
    ReLUCmplx,
    BatchNormCmplx,
    MaxPoolCmplx
)
from igcn.cmplx import new_cmplx, magnitude, cmplx
from quicktorch.utils import train
from quicktorch.models import Model
from quicktorch.data import mnist
import matplotlib.pyplot as plt


class ModulatedCmplx(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(
            IGConvCmplx(1, 32, 7, no_g=4, padding=3),
            MaxPoolCmplx(kernel_size=3),
            BatchNormCmplx(),
            ReLUCmplx(inplace=True),
            IGConvCmplx(32, 64, 7, no_g=4, padding=3),
            MaxPoolCmplx(kernel_size=3),
            BatchNormCmplx(),
            ReLUCmplx(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 10)
        )
        self.features[0].name = "modcmplx_gabor"

    def forward(self, x):
        x = new_cmplx(x)
        x = self.features(x)
        x = x[0]
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def plot(self, input, output):
    global count
    with torch.no_grad():
        plt.clf()
        img = input[0][0, 0, 0]
        activation = output.clone().detach().cpu()[:, 0:1, 0:1, :, :]

        fig, ax = plt.subplots(ncols=2)

        ax[0].imshow(img, cmap='gray')
        quiver_plot(activation, ax[1])

        for axi in ax:
            axi.set_xticks([])
            axi.set_yticks([])

        fname = "figs/filters/{}_c{}".format(self.name, count)
        count = count + 1

        fig.tight_layout()
        fig.savefig(fname)

        plot_weights(self)


def plot_weights(self):
    global count
    print(f'self.ReConv.weight.size()={self.ReConv.weight.size()}')
    print(f'self.ImConv.weight.size()={self.ImConv.weight.size()}')
    conv_weight = cmplx(self.ReConv.weight, self.ImConv.weight)
    enhanced_weight = self.gabor(conv_weight)[:, :1, :1, :, :]
    conv_weight = conv_weight[:, :1, :1, :, :]
    print(f'enhanced_weight.size()={enhanced_weight.size()}')
    fig, ax = plt.subplots(ncols=2)
    quiver_plot(conv_weight, ax[0])
    quiver_plot(enhanced_weight, ax[1])

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])

    fname = "figs/filters/{}_c{}_weights".format(self.name, count)
    count = count + 1

    fig.tight_layout()
    fig.savefig(fname)


def quiver_plot(cmplx_t, ax):
    print(f'cmplx_t.size()={cmplx_t.size()}')
    h, w = cmplx_t.size(3), cmplx_t.size(4)
    x = magnitude(cmplx_t).view(h, w)
    cmplx_t = cmplx_t.view(2, h, w)
    vectors = cmplx_t / x * 2
    Y, X = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    ax.imshow(x)
    ax.quiver(X, Y, vectors[0], vectors[1])



if __name__ == "__main__":
    global count
    count = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ModulatedCmplx(name='modcmplx')
    model_type = None
    load = False
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    if len(sys.argv) > 2:
        load = bool(sys.argv[2])

    net = net.to(device)
    data, test_data, _ = mnist(batch_size=500, rotate=True)
    if load:
        net.load(name='modcmplx_epoch5')
    else:
        epochs = 5
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        train(net, data, opt=optimizer, device=device, epochs=epochs, save_best=True)

    net.features[0].register_forward_hook(plot)
    o = net(next(iter(test_data))[0])
