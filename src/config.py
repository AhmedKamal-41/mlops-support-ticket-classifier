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
RAW_DATA_FILE = RAW_DATA_DIR / "it_support_tickets.csv"

# Local (non-MLflow) model artifacts for the lightweight CLI / Streamlit demo.
# The MLflow pipeline (src/train.py) still logs to mlruns/; these files let the
# demo run end-to-end without an MLflow server.
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "classifier.pkl"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"

# MLflow configuration
# Use local file-based backend (no database required)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "support_ticket_classifier")
MODEL_NAME = os.getenv("MODEL_NAME", "support_ticket_classifier")

# MLflow server configuration (for Docker)
MLFLOW_SERVER_HOST = os.getenv("MLFLOW_SERVER_HOST", "0.0.0.0")
MLFLOW_SERVER_PORT = int(os.getenv("MLFLOW_SERVER_PORT", "5000"))

# Model configuration
# Supported labels/categories for classification.
# These are the IT service desk categories the ML model predicts. The service
# desk workflow layer (src/service_desk.py) maps each of these to a priority,
# routing team, KB article, first troubleshooting steps, and escalation rule.
LABELS = [
    "Password Reset",
    "Account Lockout",
    "Microsoft 365 / Outlook",
    "Teams / OneDrive",
    "Network / Wi-Fi",
    "VPN",
    "Printer",
    "Hardware",
    "Software Installation",
    "Access Request",
    "Shared Folder",
    "Security / Phishing",
    "New Hire Setup",
    "Offboarding",
    "Escalation Required",
]

# Training configuration
TRAIN_SIZE = 0.7  # 70% of data for training
VAL_SIZE = 0.15   # 15% of data for validation
TEST_SIZE = 0.15  # 15% of data for testing

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Prometheus metrics configuration
METRICS_ENDPOINT = "/metrics"
