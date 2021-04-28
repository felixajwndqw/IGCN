import torch
import torch.nn as nn


class Scale(nn.Module):
    init_values = {
        'g': {
            'a1': 1.1243270635604858,
            'b1': 0.1694217026233673,
            'a2': 1.3301806449890137,
            'b2': 0.20295464992523193,
        },
        'r': {
            'a1': 0.9545445442199707,
            'b1': 0.4929998517036438,
            'a2': 1.2766683101654053,
            'b2': 0.4387783408164978,
        }
    }

    def __init__(self, n_channels, out_channels=None, method='arcsinh'):
        super().__init__()
        self.n_channels = n_channels
        if out_channels is None:
            self.out_channels = n_channels
        else:
            self.out_channels = out_channels
        self.n_scaling = self.out_channels // n_channels
        self.a1 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.b1 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.a2 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.b2 = nn.Parameter(data=torch.Tensor(1, self.n_scaling, n_channels, 1, 1))
        self.band_order = ['g', 'r']
        self.init_params()

    def init_params(self):
        for j in range(self.n_scaling):
            for i in range(self.n_channels):
                # self.a1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a1']), torch.tensor(.5))
                # self.b1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b1']), torch.tensor(.5))
                # self.a2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a2']), torch.tensor(.5))
                # self.b2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b2']), torch.tensor(.5))
                self.a1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a1']), torch.tensor(.5)) / (j + 1)
                self.b1.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b1']), torch.tensor(.5)) / (j + 1)
                self.a2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a2']), torch.tensor(.5)) / (j + 1)
                self.b2.data[0, j, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b2']), torch.tensor(.5)) / (j + 1)

    def forward(self, x):
        xs = x.size()
        x = x.unsqueeze(1)
        x = torch.arcsinh(self.a1 * x + self.b1)
        x = torch.sigmoid(self.a2 * x + self.b2)
        x = x.view(xs[0], self.out_channels, xs[2], xs[3])
        # print(
        #     x.size(),
        #     [
        #         (self.a1.data[0, 0, 0, 0, 0].item(), self.a2.data[0, 0, 0, 0, 0].item()),
        #         (self.a1.data[0, 1, 0, 0, 0].item(), self.a2.data[0, 1, 0, 0, 0].item()),
        #         (self.a1.data[0, 2, 0, 0, 0].item(), self.a2.data[0, 2, 0, 0, 0].item()),
        #         (self.a1.data[0, 3, 0, 0, 0].item(), self.a2.data[0, 3, 0, 0, 0].item())
        #     ],
        # )#, self.b1.data, self.b2.data)
        return x


class ScaleParallel(nn.Module):
    init_values = {
        'g': {
            'a1': 1.1243270635604858,
            'b1': 0.1694217026233673,
            'a2': 1.3301806449890137,
            'b2': 0.20295464992523193,
        },
        'r': {
            'a1': 0.9545445442199707,
            'b1': 0.4929998517036438,
            'a2': 1.2766683101654053,
            'b2': 0.4387783408164978,
        }
    }

    def __init__(self, n_channels, method='arcsinh'):
        super().__init__()
        self.n_channels = n_channels
        self.n_scaling = 2
        self.a1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.a2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.band_order = ['g', 'r']
        self.init_params()

    def init_params(self):
        for i in range(self.n_channels):
            self.a1.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a1']), torch.tensor(.5))
            self.b1.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b1']), torch.tensor(.5))
            self.a2.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['a2']), torch.tensor(.5))
            self.b2.data[0, i, 0, 0] = torch.normal(torch.tensor(self.init_values[self.band_order[i]]['b2']), torch.tensor(.5))

    def forward(self, x):
        scaled = torch.arcsinh(self.a1 * x + self.b1)
        scaled = torch.sigmoid(self.a2 * scaled + self.b2)
        return torch.cat([x, scaled], 1)
