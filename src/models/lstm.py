import torch.nn as nn

from datasets import unslide_data

NUM_FEATURES = 9

class LSTM(nn.Module):
    def __init__(self, config, window_size):
        super(LSTM, self).__init__()
        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size=config["input_size"], 
            hidden_size=config["hidden_size"], 
            num_layers=config["hidden_layers"], 
            batch_first=True
        )
        self.fc = nn.Linear(config["hidden_size"], config["output_size"])


    def forward(self, x):
        out, (hidden_state, cell_state) = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)
