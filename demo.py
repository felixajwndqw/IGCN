import torch
import torch.nn as nn
import torch.optim as optim
from igcn import IGConv
from torchvision import datasets, transforms


class IGCN(nn.Module):
    def __init__(self):
        super(IGCN, self).__init__()
        self.features = nn.Sequential(
            IGConv(1, 32, 9, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            IGConv(32, 96, 7, stride=2, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
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
        

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=512, shuffle=True)

example = iter(train_loader).next()
model = IGCN().cuda()
test_out = model(example[0].cuda())


# optimizer = optim.SGD(model.parameters(), lr=0.01)
