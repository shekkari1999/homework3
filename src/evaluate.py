import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def calculate_accuracy(predictions, true_labels, threshold=0.5):
    """
    Calculate accuracy given predictions and true labels.
    
    Args:
        predictions (torch.Tensor or np.ndarray): Model predictions (probabilities)
        true_labels (torch.Tensor or np.ndarray): True binary labels
        threshold (float): Threshold for converting probabilities to binary predictions
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    # Flatten if needed
    predictions = predictions.flatten()
    true_labels = true_labels.flatten()
    
    # Convert probabilities to binary predictions
    binary_predictions = (predictions >= threshold).astype(int)
    true_labels = true_labels.astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, binary_predictions)
    
    return accuracy


def calculate_f1_macro(predictions, true_labels, threshold=0.5):
    """
    Calculate F1 score (macro) given predictions and true labels.
    
    Args:
        predictions (torch.Tensor or np.ndarray): Model predictions (probabilities)
        true_labels (torch.Tensor or np.ndarray): True binary labels
        threshold (float): Threshold for converting probabilities to binary predictions
    
    Returns:
        float: F1 score (macro) between 0 and 1
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().detach().numpy()
    
    # Flatten if needed
    predictions = predictions.flatten()
    true_labels = true_labels.flatten()
    
    # Convert probabilities to binary predictions
    binary_predictions = (predictions >= threshold).astype(int)
    true_labels = true_labels.astype(int)
    
    # Calculate F1 score (macro)
    f1 = f1_score(true_labels, binary_predictions, average='macro', zero_division=0)
    
    return f1


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate a model on a dataloader and return accuracy and F1 score.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data
        device (torch.device): Device to run evaluation on
        threshold (float): Threshold for binary classification
    
    Returns:
        tuple: (accuracy, f1_score) as floats
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Get predictions
            preds = model(xb)
            
            # Store predictions and labels
            all_predictions.append(preds.cpu())
            all_labels.append(yb.float().unsqueeze(1).cpu() if yb.dim() == 1 else yb.float().cpu())
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    accuracy = calculate_accuracy(all_predictions, all_labels, threshold)
    f1 = calculate_f1_macro(all_predictions, all_labels, threshold)
    
    return accuracy, f1
