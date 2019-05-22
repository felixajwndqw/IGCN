import torch.nn as nn
from quicktorch.models import Model
from igcn import IGConv


class IGCN(Model):
    def __init__(self, no_g=4, model_name="default", rot_pool=False, dset="mnist", max_gabor=False):
        self.name = "igcn_" + model_name + "_" + dset
        super(IGCN, self).__init__()
        self.create_feature_block(no_g, model_name, rot_pool, dset, max_gabor)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 256, 1024),
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

    def create_feature_block(self, no_g, model_name, rot_pool, dset, max_gabor):
        modules = []
        if dset == "mnist":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(1, 32, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(64, 128, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(128, 256, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(1, 32, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(64, 128, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(128, 256, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(1, 32, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(64, 128, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(128, 256, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(1, 32, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(64, 128, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(128, 256, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
        if dset == "cifar":
            if model_name == "default" or model_name == "3":
                modules = [
                    IGConv(1, 8, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 3, rot_pool=rot_pool, padding=1, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "5":
                modules = [
                    IGConv(1, 8, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 5, rot_pool=rot_pool, padding=2, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "7":
                modules = [
                    IGConv(1, 8, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 7, rot_pool=rot_pool, padding=3, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
            if model_name == "9":
                modules = [
                    IGConv(1, 8, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor, pool_stride=2),
                    nn.ReLU(inplace=True),
                    IGConv(8, 16, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(16, 32, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True),
                    IGConv(32, 64, 9, rot_pool=rot_pool, padding=4, no_g=no_g, max_gabor=max_gabor),
                    nn.ReLU(inplace=True)
                ]
        self.features = nn.Sequential(*modules)