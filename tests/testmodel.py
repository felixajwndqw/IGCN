import torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv


class IGCNSmall(Model):
    def __init__(self, no_g=1, **kwargs):
        super(IGCNSmall, self).__init__(**kwargs)
        self.features = nn.Sequential(
            IGConv(1, 2, 3, no_g=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1682, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
