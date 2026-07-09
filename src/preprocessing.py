"""
Text preprocessing utilities for support ticket classification.

This module provides functions to preprocess text data and create
TF-IDF vectorizers for feature extraction.
"""

import re
from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text: str) -> str:
    """
    Preprocess a single text string.
    
    This function performs basic text cleaning:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove extra whitespace
    
    Args:
        text: Input text string.
    
    Returns:
        Preprocessed text string.
    
    Example:
        >>> preprocess_text("  I Was DOUBLE Charged!  ")
        'i was double charged!'
    """
    if not isinstance(text, str):
        # Handle non-string inputs (e.g., NaN)
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    # Remove extra whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text


def preprocess_texts(texts: Union[List[str], np.ndarray, pd.Series]) -> List[str]:
    """
    Preprocess a list of text strings.
    
    Args:
        texts: List, array, or Series of text strings.
    
    Returns:
        List of preprocessed text strings.
    
    Example:
        >>> texts = ["  Hello World  ", "TEST"]
        >>> preprocess_texts(texts)
        ['hello world', 'test']
    """
    return [preprocess_text(str(text)) for text in texts]


def fit_vectorizer(
    train_texts: List[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95
) -> TfidfVectorizer:
    """
    Create and fit a TF-IDF vectorizer on training texts.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text into
    numerical features that can be used by machine learning models.
    
    Args:
        train_texts: List of preprocessed training text strings.
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Range of n-grams to extract (1,2) means unigrams and bigrams.
        min_df: Minimum document frequency (ignore terms that appear in fewer docs).
        max_df: Maximum document frequency (ignore terms that appear in too many docs).
    
    Returns:
        Fitted TfidfVectorizer object.
    
    Example:
        >>> train_texts = ["i was charged", "app is crashing"]
        >>> vectorizer = fit_vectorizer(train_texts)
        >>> features = vectorizer.transform(train_texts)
    """
    # Create the vectorizer with specified parameters
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'  # Remove common English stop words
    )
    
    # Fit the vectorizer on training texts
    # This learns the vocabulary and IDF values
    vectorizer.fit(train_texts)
    
    print(f"Fitted vectorizer with {len(vectorizer.vocabulary_)} features")
    return vectorizer


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: Union[List[str], np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Transform texts into TF-IDF feature vectors using a fitted vectorizer.
    
    Args:
        vectorizer: Fitted TfidfVectorizer object.
        texts: List, array, or Series of text strings (should be preprocessed).
    
    Returns:
        Sparse matrix of TF-IDF features (can be converted to dense array if needed).
    
    Example:
        >>> vectorizer = fit_vectorizer(train_texts)
        >>> test_features = transform_texts(vectorizer, test_texts)
    """
    # Convert to list if needed
    if not isinstance(texts, list):
        texts = list(texts)
    
    # Transform texts to TF-IDF features
    features = vectorizer.transform(texts)
    
    # Convert sparse matrix to dense array for easier handling
    # (Note: for large datasets, you might want to keep it sparse)
    return features.toarray()

