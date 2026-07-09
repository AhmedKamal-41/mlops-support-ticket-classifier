"""
Data loading and splitting utilities for the support ticket classifier.

This module provides functions to load the support ticket dataset from CSV
and split it into training, validation, and test sets.
"""

import hashlib
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_FILE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE


def load_support_ticket_data(path: str = None) -> pd.DataFrame:
    """
    Load the support ticket dataset from a CSV file.
    
    Args:
        path: Path to the CSV file. If None, uses the default path from config.
    
    Returns:
        DataFrame with columns 'text' and 'label'.
    
    Example:
        >>> df = load_support_ticket_data()
        >>> print(df.head())
    """
    if path is None:
        path = str(RAW_DATA_FILE)
    
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Validate that required columns exist
    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV file must contain columns: {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Remove any rows with missing values
    df = df.dropna(subset=['text', 'label'])
    
    # Remove duplicate rows (same text and label combination)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text', 'label'], keep='first')
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate row(s)")
    
    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} support tickets from {path}")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = None,
    val_size: float = None,
    test_size: float = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training, validation, and test sets.
    
    This function performs a two-step split:
    1. First, split into train and temp (val + test)
    2. Then, split temp into val and test
    
    Args:
        df: Input DataFrame with 'text' and 'label' columns.
        train_size: Proportion of data for training (default: from config).
        val_size: Proportion of data for validation (default: from config).
        test_size: Proportion of data for testing (default: from config).
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.
    
    Example:
        >>> df = load_support_ticket_data()
        >>> train_df, val_df, test_df = train_val_test_split(df)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    """
    # Use default values from config if not provided
    if train_size is None:
        train_size = TRAIN_SIZE
    if val_size is None:
        val_size = VAL_SIZE
    if test_size is None:
        test_size = TEST_SIZE
    
    # Validate that sizes sum to 1.0
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 0.001:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, "
            f"got {total_size}"
        )
    
    # First split: separate training set from validation+test
    # temp_size is the proportion that will be split into val and test
    temp_size = val_size + test_size
    
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=df['label']  # Stratify to maintain label distribution
    )
    
    # Second split: separate validation from test
    # Calculate the proportion of temp that should be validation
    val_proportion = val_size / temp_size
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_proportion),
        random_state=random_state,
        stratify=temp_df['label']  # Stratify to maintain label distribution
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Split dataset:")
    print(f"  Training: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a fingerprint (hash) of the dataset for reproducibility.
    
    The fingerprint is computed from:
    - Sorted text column values
    - Sorted label column values
    - Number of rows
    
    This allows tracking which exact dataset was used for training.
    
    Args:
        df: DataFrame with 'text' and 'label' columns.
    
    Returns:
        Hexadecimal hash string representing the dataset.
    
    Example:
        >>> df = load_support_ticket_data()
        >>> fingerprint = compute_dataset_fingerprint(df)
        >>> print(f"Dataset fingerprint: {fingerprint}")
    """
    # Sort by text to ensure consistent hashing regardless of row order
    df_sorted = df.sort_values(by=['text', 'label']).reset_index(drop=True)
    
    # Create a string representation of the data
    data_str = ""
    data_str += "|".join(df_sorted['text'].astype(str).tolist())
    data_str += "||"
    data_str += "|".join(df_sorted['label'].astype(str).tolist())
    data_str += "||"
    data_str += str(len(df_sorted))
    
    # Compute SHA256 hash
    fingerprint = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    return fingerprint

