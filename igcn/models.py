import torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv


class IGCN(Model):
    def __init__(self, no_g=4, model_name="default", rot_pool=False, dset="mnist"):
        self.name = "igcn_" + model_name + "_" + dset
        super(IGCN, self).__init__()
        self.create_feature_block(no_g, model_name, rot_pool, dset)
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
    
    def create_feature_block(self, no_g, model_name, rot_pool, dset):
        modules = []
        if dset == "mnist":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(1, 16, 3, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 3, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 3, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(1, 16, 5, padding=1, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 5, padding=1, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 5, padding=1, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(1, 16, 7, padding=2, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 7, padding=2, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 7, padding=2, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(1, 32, 9, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 9, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
        if dset == "cifar":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(3, 16, 3, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 3, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 3, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(3, 16, 5, padding=1, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 5, padding=1, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 5, padding=1, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(3, 16, 7, padding=2, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 7, padding=2, no_g=no_g),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 7, padding=2, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(3, 32, 9, no_g=no_g, plot=True),
                    nn.ReLU(inplace=True),
                    IGConv(32, 96, 9, no_g=no_g),
                    nn.ReLU(inplace=True)
                ]
        modules = self.add_pooling(modules, rot_pool)
        self.features = nn.Sequential(*modules)


    def add_pooling(self, modules, rot_pool):
        if rot_pool:
            pass
        else:
            pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        count = 0
        for i in range(len(modules)):
            if i % 2:
                modules.insert(i + 1 + count, pooling)
                count += 1
        return modules