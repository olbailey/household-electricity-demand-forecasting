import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_epoch(model, train_loader: DataLoader, loss_function, optimizer: optim.Adam, device: torch.device, print_interval_num=10):
    model.train()
    running_loss = 0
    running_mae = 0
    total = 0
    num_batches = len(train_loader)
    num_samples = len(train_loader.dataset)
    print_interval = max(1, num_batches // print_interval_num)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predicted = model(data).squeeze(-1)

        loss = loss_function(predicted, target)
        loss.backward()
        optimizer.step()

        # Track progress
        running_loss += loss.item()
        total += target.size(0)
        running_mae += (predicted - target).abs().mean().item()

        if batch_idx % print_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            avg_mae = running_mae / (batch_idx + 1)
            print(f"\r [{total} / {num_samples}] "
                f"Loss: {avg_loss:.3f} | Mean Absolute Error: {avg_mae:.6f}", end='')
    print()

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    total_samples = len(test_loader.dataset)
    accuracy = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs).squeeze(-1)
            accuracy += (predicted - targets).abs().sum().item()

        return accuracy / total_samples
    
def finish_training(data_dir: str, model: nn.Module):
    print()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    answer = input("\nwould you like to save the model data? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        return

    model_name = input("enter the file name for the model trained: ")
    torch.save(model, os.path.join(data_dir, model_name + ".pth"))


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