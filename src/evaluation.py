"""
Evaluation utilities for the support ticket classifier.

This module provides functions to compute classification metrics
such as accuracy, precision, recall, F1-score, and confusion matrix.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def evaluate_classification_model(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict[str, float]:
    """
    Compute classification metrics for a model's predictions.
    
    This function calculates:
    - Overall accuracy
    - Per-class precision, recall, and F1-score
    - Macro-averaged and weighted-averaged metrics
    
    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        labels: List of all possible labels (for consistent ordering).
    
    Returns:
        Dictionary containing all computed metrics.
    
    Example:
        >>> y_true = ["billing", "technical", "billing"]
        >>> y_pred = ["billing", "billing", "billing"]
        >>> metrics = evaluate_classification_model(y_true, y_pred)
        >>> print(metrics['accuracy'])
    """
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute per-class metrics
    # precision_recall_fscore_support returns:
    # - precision: per-class precision scores
    # - recall: per-class recall scores
    # - fscore: per-class F1-scores
    # - support: number of samples per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,  # Return per-class metrics
        zero_division=0  # Handle division by zero gracefully
    )
    
    # Compute macro-averaged metrics (average across classes)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average='macro',
        zero_division=0
    )
    
    # Compute weighted-averaged metrics (weighted by support)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average='weighted',
        zero_division=0
    )
    
    # Build the results dictionary
    results = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
    }
    
    # Add per-class metrics if labels are provided
    if labels is not None:
        for i, label in enumerate(labels):
            results[f'{label}_precision'] = float(precision[i])
            results[f'{label}_recall'] = float(recall[i])
            results[f'{label}_f1'] = float(f1[i])
            results[f'{label}_support'] = int(support[i])
    
    return results


def get_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> np.ndarray:
    """
    Compute the confusion matrix for classification predictions.
    
    The confusion matrix shows how many samples of each true class
    were predicted as each class. This helps identify which classes
    are being confused with each other.
    
    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        labels: List of all possible labels (for consistent ordering).
    
    Returns:
        2D numpy array representing the confusion matrix.
    
    Example:
        >>> y_true = ["billing", "technical"]
        >>> y_pred = ["billing", "billing"]
        >>> cm = get_confusion_matrix(y_true, y_pred, labels=["billing", "technical"])
        >>> print(cm)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def print_classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> None:
    """
    Print a detailed classification report.
    
    This function prints a formatted report showing per-class metrics
    and overall metrics. Useful for debugging and understanding model performance.
    
    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        labels: List of all possible labels (for consistent ordering).
    
    Example:
        >>> print_classification_report(y_true, y_pred, labels=LABELS)
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

