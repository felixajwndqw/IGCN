import torch
import torch.nn as nn
import torch.optim as optim
from igcn import IGConv
from torchvision import datasets, transforms
from quicktorch.utils import train, imshow
from quicktorch.customtransforms import MakeCategorical
from quicktorch.models import Model
from quicktorch.data import mnist, cifar


class IGCN(Model):
    def __init__(self):
        self.name = "igcn_test"
        super(IGCN, self).__init__()
        self.features = nn.Sequential(
            IGConv(1, 16, 3, no_g=8, plot=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            IGConv(16, 32, 3, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            IGConv(32, 96, 3, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        # print("INPUT Start of forward", x.size())
        x = self.features(x)
        # print("INPUT After features", x.size())
        x = x.flatten(1)
        # print("INPUT Reshaped", x.size())
        x = self.classifier(x)
        # print("INPUT Classified", x.size())
        return x


dset = 'mnist'
if dset == 'mnist':
    train_loader, test_loader, _ = mnist(batch_size=512, rotate=True)
if dset == 'cifar':
    train_loader, test_loader, _ = cifar(batch_size=512)


example = iter(train_loader).next()
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
model = IGCN().cuda()
test_out = model(example[0].cuda())
for params in model.parameters():
    print(params.size())

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameter size: " + str(total_params*32/1000000) + "M")

optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model, [train_loader, test_loader], save=False, epochs=50, opt=optimizer)

example = iter(test_loader).next()
test_out = model(example[0].cuda())
imshow(example[0], test_out, classes)

