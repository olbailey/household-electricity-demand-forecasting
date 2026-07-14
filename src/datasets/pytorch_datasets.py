import pandas as pd
import numpy as np

import torch
from torch import tensor
from torch.utils.data import Dataset, Subset, DataLoader

from datasets.sliding_window import sliding_data


class MainDataset(Dataset):
    def __init__(self, root_dir, window_size):
        self.root_dir = root_dir

        df = pd.read_parquet("data/processed/processed_data.parquet")

        data = torch.from_numpy(df.to_numpy())

        self.labels = data[window_size:, :]

        mean = data.mean(dim=0)
        std = data.std(dim=0)

        data_normalized = (data - mean) / std      
        
        sliding_window: np.array = sliding_data(data_normalized, window_size)
        self.data = torch.from_numpy(sliding_window)[:-1, :]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_data_loaders(dataset, batch_size, device, val_fraction=0.15, test_fraction=0.15):
    total_size = len(dataset)
    print(f"total data points {total_size}")

    train_end = int(total_size * (1.0 - val_fraction - test_fraction))
    print(f"train end: {train_end}")
    val_end = int(total_size * (1.0 - test_fraction))
    print(f"val end: {val_end}")

    train_dataset = Subset(dataset, range(0, train_end))
    print(len(train_dataset))
    val_dataset = Subset(dataset, range(train_end, val_end))
    print(len(val_dataset))
    test_dataset = Subset(dataset, range(val_end, total_size))
    print(len(test_dataset))

    if device.type == "cuda":
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MainDataset("", 5)

    x, y, z = get_data_loaders(dataset, 64, device)