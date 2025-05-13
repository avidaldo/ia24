import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm  # For progress bars

def train_batch(model, inputs, labels, optimizer, criterion, device):
    """
    Train the model on a single batch and return statistics.
    
    Args:
        model (torch.nn.Module): Model to train
        inputs (torch.Tensor): Input data
        labels (torch.Tensor): Target labels
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (torch.nn.Module): Loss function
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        tuple: (loss, correct_predictions, total_samples)
    """
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)  # Get index of class with highest probability
    correct_predictions = (predicted == labels).sum().item()
    total_samples = labels.size(0)
    
    return loss.item(), correct_predictions, total_samples


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for a complete epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (torch.nn.Module): Loss function
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        tuple: (avg_loss, accuracy) training metrics
    """
    model.train()  # Set model to training mode
    
    # Accumulated metrics for the entire epoch
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_total = 0
    
    # For each batch in the training DataLoader
    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        batch_loss, batch_correct, batch_total = train_batch(
            model, inputs, labels, optimizer, criterion, device
        )
        
        # Accumulate statistics
        epoch_loss_sum += batch_loss
        epoch_correct += batch_correct
        epoch_total += batch_total
    
    # Calculate final metrics
    avg_loss = epoch_loss_sum / len(dataloader)
    accuracy = 100 * epoch_correct / epoch_total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    Evaluate a PyTorch model on a given set.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data
        criterion (torch.nn.Module): Loss function
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        dict: Evaluation metrics including overall loss, accuracy, and per-class accuracy.
    """
    model.eval()  # Set model to evaluation mode
    loss_sum = 0.0
    correct = 0
    total = 0
    
    # Determine the number of classes from the dataloader\'s dataset if possible
    num_classes = None
    if hasattr(dataloader.dataset, 'classes'):
        num_classes = len(dataloader.dataset.classes)
    elif hasattr(model, 'num_classes'): # Fallback if dataset doesn\'t have classes attribute
        num_classes = model.num_classes
    
    class_correct = list(0. for i in range(num_classes)) if num_classes is not None else []
    class_total = list(0. for i in range(num_classes)) if num_classes is not None else []

    with torch.no_grad():  # Disable gradient calculation to save memory and time
        for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)

            loss_sum += batch_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate class accuracy if possible
            if num_classes is not None:
                c = (predicted == labels).squeeze()
                for i in range(len(labels)): # Use len(labels) or batch size
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
    # Calculate metrics
    avg_loss = loss_sum / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate class accuracies
    class_accuracy = {}
    if num_classes is not None:
        for i in range(num_classes):
            if class_total[i] > 0:
                # Use class names if available, otherwise use index
                class_name = dataloader.dataset.classes[i] if hasattr(dataloader.dataset, 'classes') else str(i)
                class_accuracy[class_name] = 100 * class_correct[i] / class_total[i]
            else:
                class_name = dataloader.dataset.classes[i] if hasattr(dataloader.dataset, 'classes') else str(i)
                class_accuracy[class_name] = 0.0 # Handle case where a class might not be present in the batch/set
        
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy  # Add class accuracy to results
    }


def create_dataloaders(train_set, val_set, batch_size, num_workers):
    """
    Create DataLoaders for training and validation sets.
    
    Args:
        train_set (torch.utils.data.set): Training set
        val_set (torch.utils.data.set): Validation set
        batch_size (int): Batch size
        num_workers (int): Number of workers for loading data
        
    Returns:
        tuple: (train_loader, val_loader) DataLoaders
    """
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def log_epoch_stats(epoch, epochs, train_loss, train_acc, val_loss, val_acc):
    """
    Print epoch statistics.
    
    Args:
        epoch (int): Current epoch
        epochs (int): Total number of epochs
        train_loss (float): Training loss
        train_acc (float): Training accuracy
        val_loss (float): Validation loss
        val_acc (float): Validation accuracy
    """
    print(f'Epoch {epoch+1}/{epochs} completed: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


def train_model(model, train_set, val_set, batch_size, criterion, optimizer, 
                epochs=5, device='cpu', num_workers=2):
    """
    Train a PyTorch model, evaluate on validation per epoch, and log metrics.

    Creates DataLoaders internally from the provided sets to experiment
    with different batch sizes.

    Args:
        model (torch.nn.Module): The model to train
        train_set (torch.utils.data.set): Training set
        val_set (torch.utils.data.set): Validation set
        batch_size (int): Batch size for DataLoaders
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        epochs (int): Number of epochs to train
        device (str): Device to use ('cpu' or 'cuda')
        num_workers (int): Number of workers for DataLoaders

    Returns:
        dict: A dictionary containing lists of losses and accuracies
              for training and validation per epoch
    """
    # Lists to store metrics per epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    train_loader, val_loader = create_dataloaders(
        train_set, val_set, batch_size, num_workers
    )

    model = model.to(device)

    print(f"Starting training on '{device}' for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()

    for epoch in range(epochs):
        # Train for a complete epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation (at the end of each epoch)
        val_results = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Display statistics
        log_epoch_stats(epoch, epochs, train_loss, train_acc, val_loss, val_acc)

    # Completion message
    print("\nTraining completed after finishing all epochs.")

    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Return training and validation metrics as a dictionary
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


def train_final_model(model, full_train_set, batch_size, criterion, optimizer, 
                      epochs=5, device='cpu', num_workers=2):
    """
    Train a PyTorch model on the entire training dataset (including validation data) 
    without performing validation checks during training. Intended for the final 
    training phase before test set evaluation.

    Args:
        model (torch.nn.Module): The model to train
        full_train_set (torch.utils.data.Dataset): The complete training dataset
        batch_size (int): Batch size for DataLoader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        epochs (int): Number of epochs to train
        device (str): Device to use ('cpu' or 'cuda')
        num_workers (int): Number of workers for DataLoader

    Returns:
        dict: A dictionary containing the list of training losses and accuracies per epoch
    """
    # Lists to store metrics per epoch
    train_losses = []
    train_accs = []

    full_train_loader = torch.utils.data.DataLoader(
        full_train_set, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle is important for training
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    model = model.to(device)

    print(f"Starting final training on '{device}' for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()

    for epoch in range(epochs):
        # Train for a complete epoch
        # Re-use train_epoch as it only needs the loader
        train_loss, train_acc = train_epoch(
            model, full_train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Log only training stats
        print(f"Epoch {epoch+1}/{epochs} => "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    print("\nFinal training completed.")
    elapsed_time = time.time() - start_time
    print(f"Total final training time: {elapsed_time:.2f} seconds")

    # Return training metrics as a dictionary
    return {
        'train_losses': train_losses,
        'train_accs': train_accs
    }