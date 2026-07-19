import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP
from datasets.pytorch_datasets import MainDataset, get_data_loaders

from .utils import train_epoch, evaluate, finish_training
from .utils import plot_predictions

MODEL_DATA_DIR = "outputs/models"

NUM_FEATURES = 9
WINDOW_SIZE = 5
BATCH_SIZE = 512
LOSS = 0.001

ENABLE_MODEL_SAVING = False
SHOWING_GRAPHS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(NUM_FEATURES, WINDOW_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LOSS)

electricity_dataset = MainDataset("data/processed/processed_data.parquet", WINDOW_SIZE)

train_loader, val_loader, test_loader = get_data_loaders(electricity_dataset, BATCH_SIZE, device)

num_epochs = 100

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}")
        train_epoch(model, train_loader, loss_function, optimizer, device)
        accuracy = evaluate(model, val_loader, device)
        print(f"Validation MAE: {accuracy:.6f}")
        if SHOWING_GRAPHS:
            try:
                x = int(input("How many points would you like to plot? "))
                plot_predictions(model, val_loader, device, num_points=x)
            except ValueError:
                pass
except KeyboardInterrupt:
    if ENABLE_MODEL_SAVING:
        finish_training(MODEL_DATA_DIR, model)