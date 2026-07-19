import pandas as pd
import numpy as np

import torch
from torch import tensor
from torch.utils.data import Dataset, Subset, DataLoader

from datasets.sliding_window import sliding_data
from utils.print import tensor_to_decimal_point

class MainDataset(Dataset):
    def __init__(self, root_dir, window_size):
        self.window_size = window_size

        df = pd.read_parquet(root_dir)

        data = torch.from_numpy(df.to_numpy()).float()

        self.labels = self._format_labels(data)

        mean = data.mean(dim=0)
        std = data.std(dim=0)
        print(f"mean: {tensor_to_decimal_point(mean, dp=3)}")
        print(f"std:  {tensor_to_decimal_point(std, dp=3)}")

        data_normalized = (data - mean) / std      
        
        sliding_window: np.array = sliding_data(data_normalized, self.window_size)
        self.data = torch.from_numpy(sliding_window).float()[:-1, :]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def _format_labels(self, data: torch.Tensor) -> torch.Tensor:
        # For getting the global active power values for each data window
        # labels = data[self.window_size:, 0]


        # For getting the differences between each global active power each time step
        current_data_values = data[(self.window_size-1):, 0].clone()
        previous_data_values = torch.cat((torch.tensor([0]), current_data_values[:-1]))
        
        labels = current_data_values - previous_data_values

        return labels

    
def get_data_loaders(dataset, batch_size, device, val_fraction=0.15, test_fraction=0.15):
    total_size = len(dataset)

    train_end = int(total_size * (1.0 - val_fraction - test_fraction))
    val_end = int(total_size * (1.0 - test_fraction))

    train_dataset = Subset(dataset, range(0, train_end))
    val_dataset = Subset(dataset, range(train_end, val_end))
    test_dataset = Subset(dataset, range(val_end, total_size))

    if device.type == "cuda":
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MainDataset("", 5)

    x, y, z = get_data_loaders(dataset, 64, device)