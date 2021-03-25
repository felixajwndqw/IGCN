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

    def __init__(self, n_channels, method='arcsinh'):
        super().__init__()
        self.n_channels = n_channels
        self.a1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b1 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.a2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.b2 = nn.Parameter(data=torch.Tensor(1, n_channels, 1, 1))
        self.band_order = ['g', 'r']
        self.init_params()

    def init_params(self):
        for i in range(self.n_channels):
            print(i)
            self.a1.data[0, i, 0, 0] = self.init_values[self.band_order[i]]['a1']
            self.b1.data[0, i, 0, 0] = self.init_values[self.band_order[i]]['b1']
            self.a2.data[0, i, 0, 0] = self.init_values[self.band_order[i]]['a2']
            self.b2.data[0, i, 0, 0] = self.init_values[self.band_order[i]]['b2']

    def forward(self, x):
        x = torch.arcsinh(self.a1 * x + self.b1)
        x = torch.sigmoid(self.a2 * x + self.b2)
        return x
