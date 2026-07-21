import yaml

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP, LSTM
from datasets.pytorch_datasets import MainDataset, get_data_loaders

from .utils import train_epoch, evaluate, finish_training, EarlyStopping
from .utils import show_graph



with open("configs/lstm.yaml", 'r') as file:
    configs = yaml.safe_load(file)

MODEL_DATA_DIR = configs["data"]["model_dir"]
MODEL_TEMP_DATA_DIR = "outputs/models/temp"

NUM_FEATURES = configs["training"]["num_features"]
WINDOW_SIZE = configs["training"]["window_size"]
BATCH_SIZE = configs["training"]["batch_size"]
LEARNING_RATE = configs["training"]["learning_rate"]

ENABLE_MODEL_SAVING = False
OVERIDE_SHOWING_GRAPHS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if configs["model"]["type"] == "MultiLayerPerceptron":
    model = MLP(configs["model"])
elif configs["model"]["type"] == "LSTM":
    model = LSTM(configs["model"], WINDOW_SIZE)


loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
schedular_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
schedular_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)

electricity_dataset = MainDataset("data/processed/processed_data.parquet", WINDOW_SIZE)

train_loader, val_loader, test_loader = get_data_loaders(electricity_dataset, BATCH_SIZE, device)

num_epochs = 50

early_stopping = EarlyStopping(
    MODEL_TEMP_DATA_DIR, 
    patience=configs["training"]["patience"], 
    min_delta=configs["training"]["min_delta"]
)

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}")
        train_epoch(model, train_loader, loss_function, optimizer, device)

        mae, rmse = evaluate(model, val_loader, device)
        print(f"Validation MAE: {mae:.6f}, RMSE: {rmse:.6f}, lr: {schedular_plateau.get_last_lr()[0]:.6f}")

        model = early_stopping.update(model, mae)

        schedular_plateau.step(mae)

        show_graph(model, val_loader, device, overide_show=OVERIDE_SHOWING_GRAPHS)

        if early_stopping.stopped:
            show_graph(model, val_loader, device)
            finish_training(MODEL_DATA_DIR, model)
            break
        
except KeyboardInterrupt:
    if ENABLE_MODEL_SAVING:
        finish_training(MODEL_DATA_DIR, model)