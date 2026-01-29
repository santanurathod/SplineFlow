import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from torchdyn.core import NeuralODE

import matplotlib.pyplot as plt
import pickle


import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3) -> None:
#         super().__init__()
#         layers = []
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.Tanh())
#         for i in range(num_layers-1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_dim, output_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        for m in self.hidden_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.tanh(self.in_layer(x))
        for layer in self.hidden_layers:
            h = h + torch.tanh(layer(h))   # residual block
        return self.out_layer(h)


# class MLP(nn.Module):
#     def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, num_layers=None):
#         super().__init__()
#         layers = []
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_dim, output_dim))  # no final activation
#         self.net = nn.Sequential(*layers)

#         for m in self.net:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
#                 nn.init.zeros_(m.bias)

#     def forward(self, inp):
#         return self.net(inp)