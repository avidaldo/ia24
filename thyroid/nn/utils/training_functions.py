import torch
import time
from utils.scoring import custom_recall
import numpy as np


def train_epoch(model, dataloader, optimizer, criterion, le):
    model.train()
    epoch_loss_sum = 0.0
    
    # Se acumulan todas las predicciones y etiquetas reales de la época para calcular 
    # Esto es necesario para calcular métricas correctas cuando los batches tienen tamaños diferentes
    all_predictions = []
    all_labels = []
    
    for inputs, labels in dataloader:
        all_labels.extend(labels.cpu().numpy())
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)

        epoch_loss_sum += loss.item()
        all_predictions.extend(predicted.cpu().numpy())
    
    # Convertir a arrays numpy para cálculos eficientes
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calcular métricas sobre todo el conjunto de la época
    avg_loss = epoch_loss_sum / len(dataloader)
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    recall = custom_recall(all_labels, all_predictions, le)
    
    return avg_loss, accuracy, recall


def evaluate_model(model, dataloader, criterion, le):
    model.eval()  # Modo evaluación (desactiva dropout, etc.)
    loss_sum = 0.0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            all_labels.extend(labels.cpu().numpy())
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            
            loss_sum += batch_loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    avg_loss = loss_sum / len(dataloader)
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    recall = custom_recall(all_labels, all_predictions, le)
    
    return avg_loss, accuracy, recall


def log_epoch_stats(epoch, epochs, train_loss, train_acc, val_loss, val_acc, train_recall, val_recall, log_every=None):
    if log_every is not None and epoch % log_every == 0:
        print(f'Epoch {epoch+1}/{epochs} completed: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
            f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}')


def train_model(model, train_loader, val_loader, batch_size, criterion, optimizer, le,  epochs, log_every=None):
    # Lists to store metrics per epoch
    train_losses = []
    train_accs = []
    train_recalls = []
    val_losses = []
    val_accs = []
    val_recalls = []

    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc, train_recall = train_epoch(
            model, train_loader, optimizer, criterion, le
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_recalls.append(train_recall)
        
        # Validation (at the end of each epoch)
        val_loss, val_acc, val_recall = evaluate_model(model, val_loader, criterion, le)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recalls.append(val_recall)

        # Display statistics
        log_epoch_stats(epoch, epochs, train_loss, train_acc, val_loss, val_acc, train_recall, val_recall, log_every=log_every)

    print("\nTraining completed after finishing all epochs.")

    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Return training and validation metrics as a dictionary
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_recalls': train_recalls,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_recalls': val_recalls
    }


