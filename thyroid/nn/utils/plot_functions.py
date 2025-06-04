import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """Plot training and validation metrics over epochs"""
    epochs_range = range(1, len(metrics['train_losses']) + 1)
    
    # Create subplots
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(epochs_range, metrics['train_losses'], 'b-', label='Training Loss')
    axes[0].plot(epochs_range, metrics['val_losses'], 'r-', label='Validation Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Accuracy
    axes[1].plot(epochs_range, metrics['train_accs'], 'b-', label='Training Accuracy')
    axes[1].plot(epochs_range, metrics['val_accs'], 'r-', label='Validation Accuracy')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Recall
    axes[2].plot(epochs_range, metrics['train_recalls'], 'b-', label='Training Recall')
    axes[2].plot(epochs_range, metrics['val_recalls'], 'r-', label='Validation Recall')
    axes[2].set_title('Positive Recall Over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Positive Recall (Hyperthyroidism & Hypothyroidism)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()