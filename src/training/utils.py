import os
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from tqdm.auto import tqdm


def train_epoch(model, train_loader: DataLoader, loss_function, optimizer: optim.Adam, device: torch.device, print_interval_num=10):
    model.train()
    running_loss = 0
    running_mae = 0
    total = 0
    num_batches = len(train_loader)
    print_interval = max(1, num_batches // print_interval_num)

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    # for batch_idx, (data, target) in enumerate(train_loader):
    batch_idx = 0
    for data, target in train_progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predicted = model(data).squeeze(-1)

        loss = loss_function(predicted, target)
        loss.backward()
        optimizer.step()

        # Track progress
        running_loss += loss.item()
        running_mae += (predicted - target).abs().mean().item()

        if batch_idx % print_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            avg_mae = running_mae / (batch_idx + 1)
            
            train_progress_bar.set_postfix(loss=f"{avg_loss:.3f}", MAE=f"{avg_mae:.6f}")
        batch_idx += 1

def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()

    rmse_metric = MeanSquaredError(squared=False)
    mae_metric = MeanAbsoluteError()

    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Validation", unit="batch")

        for inputs, targets in val_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs).squeeze(-1)

            rmse_metric.update(predicted, targets)
            mae_metric.update(predicted, targets)

    mae = mae_metric.compute().item()
    rmse = rmse_metric.compute().item()

    return mae, rmse
    
class EarlyStopping:
    def __init__(self, temp_model_dir, patience, min_delta, restore_best_weights=True):
        self.current_best = math.inf
        self.temp_model_dir = temp_model_dir
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.count = 0
        self.stopped = False

    def update(self, model, value) -> nn.Module:
        if value < self.current_best and self.current_best - value > self.min_delta:
            self.save_model(model)
            print(f"New best, saving model...")
            self.current_best = value
            self.count = 0
        elif self.count < self.patience:
            self.count += 1 
            print(f"Model has regressed, current count: {self.count}/{self.patience}, Value Delta: {self.current_best - value:.8f}")
        elif self.restore_best_weights:
            model = self.restore_best_model(model)
            self.stopped = True

        return model
        
    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.temp_model_dir, "model_temp_save.pt"))

    def restore_best_model(self, model:nn.Module):
        try:
            model.load_state_dict(torch.load(os.path.join(self.temp_model_dir, "model_temp_save.pt")))
        except FileNotFoundError:
            print("ERROR! Could not find model parameter file!")

        return model
    
    
def finish_training(data_dir: str, model: nn.Module):
    print()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    answer = input("\nwould you like to save the model data? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        return

    model_name = input("enter the file name for the model trained: ")
    torch.save(model.state_dict(), os.path.join(data_dir, model_name + ".pt"))

def show_graph(model, val_loader, device, overide_show=False):
    if not overide_show:
        try:
            x = int(input("How many points would you like to plot? "))
            plot_predictions(model, val_loader, device, num_points=x)
        except ValueError:
            pass

def plot_predictions(model, loader: DataLoader, device: torch.device, num_points: int = None, title: str = "Predicted vs Actual"):
    """
    Runs the model over a DataLoader and plots predicted vs actual values.

    num_points: if set, only plots the first N points (useful for zooming in
                on long time series where plotting everything is unreadable).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(-1)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    if num_points is not None and num_points > 0:
        preds = preds[:num_points]
        targets = targets[:num_points]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top plot: predicted vs actual over "time" (i.e. sample index)
    axes[0].plot(targets, label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(preds, label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Value')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bottom plot: scatter of predicted vs actual (perfect predictions fall on y=x line)
    axes[1].scatter(targets, preds, alpha=0.4, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect prediction')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Predicted vs Actual (scatter)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return preds, targets