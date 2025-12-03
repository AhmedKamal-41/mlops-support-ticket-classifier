"""
Inference utilities for loading models and making predictions.

This module provides functions to load the production model from MLflow
and use it to make predictions on new support ticket texts.
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from typing import List, Tuple, Optional
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME
from src.preprocessing import preprocess_texts


# Global variables to cache the loaded model and vectorizer
_model = None
_vectorizer = None


def load_production_model():
    """
    Load the Production stage model from MLflow Model Registry.
    
    This function:
    1. Connects to MLflow tracking server
    2. Retrieves the Production stage model
    3. Loads the model and associated vectorizer artifact
    4. Caches them in global variables for reuse
    
    Returns:
        Tuple of (model, vectorizer) objects.
    
    Raises:
        Exception: If the model cannot be loaded or doesn't exist.
    
    Example:
        >>> model, vectorizer = load_production_model()
        >>> predictions = model.predict(features)
    """
    global _model, _vectorizer
    
    # If already loaded, return cached versions
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer
    
    print("Loading production model from MLflow...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Get the MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Try to get the Production stage model first
        try:
            model_version = client.get_latest_versions(
                MODEL_NAME,
                stages=["Production"]
            )[0]
            print(f"Found Production model: version {model_version.version}")
        except (IndexError, Exception):
            # If no Production model exists, get the latest version
            print("No Production model found, using latest version...")
            model_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1]
            print(f"Using latest model: version {model_version.version}")
        
        # Load the model from MLflow
        model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
        print(f"Loading model from: {model_uri}")
        _model = mlflow.sklearn.load_model(model_uri)
        
        # Load the vectorizer artifact
        # The vectorizer is stored as an artifact in the run
        run_id = model_version.run_id
        print(f"Loading vectorizer from run: {run_id}")
        
        # Download the vectorizer artifact
        # The vectorizer is saved in the "vectorizer" artifact directory
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="vectorizer/vectorizer.pkl"
        )
        
        # Load the vectorizer
        _vectorizer = joblib.load(artifact_path)
        print("Model and vectorizer loaded successfully!")
        
        return _model, _vectorizer
        
    except Exception as e:
        error_msg = (
            f"Failed to load production model: {e}\n"
            f"Make sure you have trained a model first by running: python -m src.train"
        )
        raise Exception(error_msg) from e


def predict_tickets(
    texts: List[str],
    return_proba: bool = False
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Predict categories for a list of support ticket texts.
    
    This function:
    1. Preprocesses the input texts
    2. Transforms them using the vectorizer
    3. Makes predictions using the model
    4. Optionally returns prediction probabilities
    
    Args:
        texts: List of support ticket text strings.
        return_proba: If True, also return prediction probabilities.
    
    Returns:
        Tuple of (predictions, probabilities):
        - predictions: List of predicted category labels
        - probabilities: Optional numpy array of prediction probabilities
          (shape: [n_samples, n_classes])
    
    Example:
        >>> tickets = ["I was double charged", "App keeps crashing"]
        >>> predictions, probas = predict_tickets(tickets, return_proba=True)
        >>> print(predictions)
        ['billing', 'technical']
    """
    # Load model and vectorizer if not already loaded
    model, vectorizer = load_production_model()
    
    # Preprocess the texts
    preprocessed_texts = preprocess_texts(texts)
    
    # Transform texts to feature vectors
    # Note: transform_texts returns a dense array, but vectorizer.transform
    # returns a sparse matrix. We'll use vectorizer directly for consistency.
    features = vectorizer.transform(preprocessed_texts)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Get probabilities if requested
    probabilities = None
    if return_proba:
        probabilities = model.predict_proba(features)
    
    return predictions.tolist(), probabilities


def get_model_info() -> dict:
    """
    Get information about the currently loaded model.
    
    Returns:
        Dictionary with model information (version, run_id, etc.).
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Try to get Production model
        try:
            model_version = client.get_latest_versions(
                MODEL_NAME,
                stages=["Production"]
            )[0]
        except (IndexError, Exception):
            model_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1]
        
        return {
            "model_name": MODEL_NAME,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "source": model_version.source
        }
    except Exception as e:
        return {"error": str(e)}

