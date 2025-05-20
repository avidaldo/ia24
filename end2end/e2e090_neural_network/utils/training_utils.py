import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt



def _print_epoch_stats(epoch, num_epochs, epoch_duration, train_rmse, val_rmse):
    print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_duration:.2f}s | Train RMSE: {train_rmse:.0f} | Val RMSE: {val_rmse:.0f}")


def _train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_train_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * batch_X.size(0)
        total_samples += batch_X.size(0)
    
    return running_train_loss / total_samples


def evaluate_model(model, dataloader, criterion, y_scaler, device='cpu'):
    """
    Evaluates a PyTorch model on a given dataset and returns denormalized RMSE.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        criterion (torch.nn.Module): Loss function (e.g., MSE).
        y_scaler (sklearn.preprocessing.StandardScaler): Scaler used for y variable.
        device (str): Device to use ('cpu' or 'cuda').
        
    Returns:
        float: Denormalized RMSE.
        float: Average normalized loss for the dataset.
    """
    model.eval()
    all_preds_np = []
    all_labels_np = []
    running_loss_normalized = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            running_loss_normalized += loss.item() * batch_X.size(0)

            all_preds_np.append(outputs.cpu().numpy())
            all_labels_np.append(batch_y.cpu().numpy())
    
    avg_loss_normalized = running_loss_normalized / len(dataloader.dataset)
    
    preds_np = np.concatenate(all_preds_np)
    labels_np = np.concatenate(all_labels_np)
    
    if preds_np.ndim == 1:
        preds_np = preds_np.reshape(-1, 1)
    if labels_np.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)
        
    preds_denorm = y_scaler.inverse_transform(preds_np)
    labels_denorm = y_scaler.inverse_transform(labels_np)
    
    rmse_denorm = root_mean_squared_error(labels_denorm, preds_denorm)
    
    return rmse_denorm, avg_loss_normalized

def collect_epoch_metrics(model, train_loader, val_loader, criterion, y_scaler, device):
    """
    Collects training and validation metrics for an epoch.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        y_scaler (sklearn.preprocessing.StandardScaler): Scaler for the target variable
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        tuple: (train_rmse, val_rmse)
    """
    
    model.eval()
    train_rmse, _ = evaluate_model(model, train_loader, criterion, y_scaler, device)
    val_rmse, _ = evaluate_model(model, val_loader, criterion, y_scaler, device)
    
    return train_rmse, val_rmse



def train_model_detailed(model, train_loader, val_loader, criterion, optimizer, 
                         y_scaler, num_epochs, device, print_every=10):
    """
    Trains a PyTorch model, evaluates on training and validation sets each epoch,
    prints detailed metrics, and returns a history.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer.
        y_scaler (sklearn.preprocessing.StandardScaler): Scaler for the target variable.
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ('cpu' or 'cuda').
        print_every (int): How often to print epoch metrics.

    Returns:
        dict: A dictionary containing lists of metrics per epoch:
              'train_rmse', 'val_rmse'
    """
    history = {
        'train_rmse': [],      # Denormalized RMSE on training set
        'val_rmse': [],        # Denormalized RMSE on validation set
    }
    
    model.to(device)
    
    print(f"Starting detailed training on '{device}' for {num_epochs} epochs...")
    start_time_total = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        _train_one_epoch(model, train_loader, criterion, optimizer, device)

        train_rmse, val_rmse = collect_epoch_metrics(
            model, train_loader, val_loader, criterion, y_scaler, device
        )
        
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        
        epoch_duration = time.time() - epoch_start_time
        
        if (epoch + 1) % print_every == 0 or epoch == num_epochs - 1:
            _print_epoch_stats(epoch, num_epochs, epoch_duration,train_rmse, val_rmse)
            
    total_training_time = time.time() - start_time_total
    print(f"\nDetailed training finished in {total_training_time:.2f} seconds.")
    return history




def plot_metrics(*metrics_list):
    """Plot multiple training metrics histories together.
    
    Args:
        *metrics_list: Variable number of metric dictionaries to plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define line styles for different metrics (all except last will use these)
    line_styles = ['--', ':', '-.', '..']  # Add more if needed
    
    for i, metrics in enumerate(metrics_list):
        suffix = f"_{i+1}" if len(metrics_list) > 1 else ""
        # Use continuous line style only for the last metrics
        style = '-' if i == len(metrics_list) - 1 else line_styles[i % len(line_styles)]
        plt.plot(metrics['train_rmse'], linestyle=style, label=f'Train RMSE{suffix}')
        plt.plot(metrics['val_rmse'], linestyle=style, label=f'Validation RMSE{suffix}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Denormalized RMSE') 
    plt.legend()
    plt.grid(True)
    plt.show()