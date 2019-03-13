import torch
import torch.nn as nn
import torch.optim as optim
from igcn import IGConv
from torchvision import datasets, transforms


class IGCN(nn.Module):
    def __init__(self):
        super(IGCN, self).__init__()
        self.features = nn.Sequential(
            IGConv(3, 32, 9, stride=3, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            IGConv(32, 96, 7, stride=2, no_g=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
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
print(example[0].size())
model = IGCN().cuda()
test_out = model(example[0])
# optimizer = optim.SGD(model.parameters(), lr=0.01)
