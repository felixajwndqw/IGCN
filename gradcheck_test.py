from torch.autograd import gradcheck
import torch
from igcn import GaborFunction
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
            nn.Linear(8, 3)
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


# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (
            torch.randn(4, 3, 10, 10, dtype=torch.double, requires_grad=True),
            torch.ones(2, 4, dtype=torch.double, requires_grad=True)
        )

gabor = GaborFunction.apply
test = gradcheck(gabor, input, eps=1e-6, atol=1e-4)
print("GaborFunction results:")
print(test)

test_img = torch.randn(4, 1, 10, 10, dtype=torch.double, requires_grad=True)

smallmodel = IGCNSmall().double()
test = gradcheck(smallmodel, test_img, eps=1e-6, atol=1e-4)
print("Small model with GCN layer results:")
print(test)