"""
Simple data drift detection for support ticket classification.

This module implements a basic drift detection mechanism that tracks
the distribution of ticket lengths (number of words) between training
data and incoming requests. Significant changes in this distribution
may indicate data drift.
"""

from typing import List, Tuple
import numpy as np
from collections import deque


class DriftDetector:
    """
    A simple drift detector that tracks ticket length distribution.
    
    This detector maintains a rolling window of recent ticket lengths
    and compares their distribution to the training data distribution.
    """
    
    def __init__(self, reference_lengths: List[int], window_size: int = 100):
        """
        Initialize the drift detector.
        
        Args:
            reference_lengths: List of ticket lengths from training data
                (used as baseline for comparison).
            window_size: Size of the rolling window for recent tickets.
        """
        self.reference_lengths = np.array(reference_lengths)
        self.window_size = window_size
        self.recent_lengths = deque(maxlen=window_size)
        
        # Compute reference statistics
        self.reference_mean = np.mean(self.reference_lengths)
        self.reference_std = np.std(self.reference_lengths)
        
        print(f"Drift detector initialized:")
        print(f"  Reference mean length: {self.reference_mean:.2f} words")
        print(f"  Reference std length: {self.reference_std:.2f} words")
        print(f"  Window size: {window_size}")
    
    def update(self, texts: List[str]) -> None:
        """
        Update the detector with new ticket texts.
        
        Args:
            texts: List of new support ticket text strings.
        """
        # Compute lengths (number of words) for each text
        lengths = [len(text.split()) for text in texts]
        
        # Add to rolling window
        self.recent_lengths.extend(lengths)
    
    def compute_drift_score(self) -> float:
        """
        Compute a drift score comparing recent tickets to training data.
        
        The drift score is computed as:
        - If we have enough recent samples (>= window_size/2):
          - Compare mean of recent lengths to reference mean
          - Normalize by reference standard deviation
          - Return absolute difference as drift score
        - Otherwise, return 0.0 (not enough data)
        
        Returns:
            Drift score (0.0 = no drift, higher = more drift).
            A score > 1.0 suggests significant drift.
        """
        if len(self.recent_lengths) < self.window_size // 2:
            # Not enough data yet
            return 0.0
        
        recent_array = np.array(list(self.recent_lengths))
        recent_mean = np.mean(recent_array)
        
        # Compute normalized difference
        # If reference_std is 0, use a small epsilon to avoid division by zero
        if self.reference_std < 1e-6:
            drift_score = abs(recent_mean - self.reference_mean)
        else:
            # Normalize by standard deviation (z-score difference)
            drift_score = abs(recent_mean - self.reference_mean) / self.reference_std
        
        return float(drift_score)
    
    def get_statistics(self) -> dict:
        """
        Get current statistics about the drift detector.
        
        Returns:
            Dictionary with reference and recent statistics.
        """
        stats = {
            "reference_mean": float(self.reference_mean),
            "reference_std": float(self.reference_std),
            "recent_count": len(self.recent_lengths),
            "window_size": self.window_size
        }
        
        if len(self.recent_lengths) > 0:
            recent_array = np.array(list(self.recent_lengths))
            stats["recent_mean"] = float(np.mean(recent_array))
            stats["recent_std"] = float(np.std(recent_array))
            stats["drift_score"] = self.compute_drift_score()
        else:
            stats["recent_mean"] = None
            stats["recent_std"] = None
            stats["drift_score"] = 0.0
        
        return stats


def compute_drift_score_on_length(
    train_texts: List[str],
    recent_texts: List[str]
) -> float:
    """
    Compute a simple drift score by comparing ticket length distributions.
    
    This is a standalone function that doesn't maintain state.
    Useful for one-off drift checks.
    
    Args:
        train_texts: List of training ticket texts (baseline).
        recent_texts: List of recent ticket texts to compare.
    
    Returns:
        Drift score (0.0 = no drift, higher = more drift).
    """
    # Compute lengths (number of words) for each text
    train_lengths = [len(text.split()) for text in train_texts]
    recent_lengths = [len(text.split()) for text in recent_texts]
    
    # Compute statistics
    train_mean = np.mean(train_lengths)
    train_std = np.std(train_lengths)
    recent_mean = np.mean(recent_lengths)
    
    # Compute normalized difference
    if train_std < 1e-6:
        drift_score = abs(recent_mean - train_mean)
    else:
        drift_score = abs(recent_mean - train_mean) / train_std
    
    return float(drift_score)

