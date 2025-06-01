from sklearn.metrics import recall_score
import numpy as np


def custom_recall(y_true, y_pred, le):
    # Get original class names from the LabelEncoder
    all_classes = le.classes_
    # Calculate recall for each class using original labels
    recalls = recall_score(le.inverse_transform(y_true), 
                          le.inverse_transform(y_pred), 
                          labels=all_classes, 
                          average=None, 
                          zero_division=0)
    # Get indices for disease classes
    disease_indices = [np.where(all_classes == cls)[0][0] 
                      for cls in ['hyperthyroid', 'hypothyroid']]
    return np.mean(recalls[disease_indices])