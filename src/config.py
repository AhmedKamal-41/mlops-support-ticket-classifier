"""
Configuration constants for the MLOps Support Ticket Classifier project.

This module contains all the configuration settings used throughout the project,
including MLflow tracking URI, experiment names, model names, and file paths.
"""

import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "support_tickets_sample.csv"

# MLflow configuration
# Use local file-based backend (no database required)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "support_ticket_classifier")
MODEL_NAME = os.getenv("MODEL_NAME", "support_ticket_classifier")

# MLflow server configuration (for Docker)
MLFLOW_SERVER_HOST = os.getenv("MLFLOW_SERVER_HOST", "0.0.0.0")
MLFLOW_SERVER_PORT = int(os.getenv("MLFLOW_SERVER_PORT", "5000"))

# Model configuration
# Supported labels/categories for classification
LABELS = ["billing", "technical", "account", "shipping", "general"]

# Training configuration
TRAIN_SIZE = 0.7  # 70% of data for training
VAL_SIZE = 0.15   # 15% of data for validation
TEST_SIZE = 0.15  # 15% of data for testing

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Prometheus metrics configuration
METRICS_ENDPOINT = "/metrics"

