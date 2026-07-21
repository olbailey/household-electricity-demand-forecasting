import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config, window_size):
        super(MLP, self).__init__()

        activation = nn.ReLU() if config["activation"] == "relu" else nn.Tanh()

        layers = []

        current_in = config["input_size"] * window_size
        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(current_in, hidden_dim))
            layers.append(activation)
            current_in = hidden_dim

        layers.append(nn.Linear(current_in, config["output_size"]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        return self.layers(x)
    