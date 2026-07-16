import os

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_features, window_size):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features * window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)
    