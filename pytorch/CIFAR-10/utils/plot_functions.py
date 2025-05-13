import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics_list, model_names, figsize=(14, 5)):
    """
    Visualize training and validation metrics for multiple models.
    
    Args:
        metrics_list (list): List of dictionaries containing metrics for each model
        model_names (list): List of model names corresponding to metrics_list
        figsize (tuple): Figure size as (width, height)
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    for metrics, name in zip(metrics_list, model_names):
        ax1.plot(metrics['train_losses'], label=f'{name} (train)')
        ax1.plot(metrics['val_losses'], label=f'{name} (val)', linestyle='--')
    
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    for metrics, name in zip(metrics_list, model_names):
        ax2.plot(metrics['train_accs'], label=f'{name} (train)')
        ax2.plot(metrics['val_accs'], label=f'{name} (val)', linestyle='--')
    
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    

def plot_class_accuracy(results_list, model_names, class_names, figsize=(12, 6)):
    """
    Visualize per-class accuracy for multiple models.
    
    Args:
        results_list (list): List of dictionaries containing class accuracy results
        model_names (list): List of model names corresponding to results_list
        class_names (list): List of class names/labels
        figsize (tuple): Figure size as (width, height)
    """
    _, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    offset = -0.4 + width/2
    
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        class_acc = [results['class_accuracy'][cls] for cls in class_names]
        ax.bar(x + offset + i*width, class_acc, width, label=name)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()