"""
FastAPI application for support ticket classification.

This module creates the FastAPI app with endpoints for:
- Health checks
- Batch prediction of support ticket categories
- Prometheus metrics exposure

The app loads the production model from MLflow on startup and
uses it to serve predictions. It also tracks data drift and
exposes metrics for monitoring.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.schemas import PredictionRequest, PredictionResponse, TicketPrediction, HealthResponse
from src.inference import load_production_model, predict_tickets, get_model_info
from src.drift_detection import DriftDetector
from src.data_loader import load_support_ticket_data, train_val_test_split
from src.preprocessing import preprocess_texts
from src.config import MLFLOW_TRACKING_URI

# Global variables to store loaded model and drift detector
model = None
vectorizer = None
drift_detector = None

# Prometheus custom metrics
drift_score_metric = Gauge(
    'support_ticket_drift_score',
    'Data drift score for support tickets (higher = more drift)',
    ['model_version']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    This function:
    - Loads the model and vectorizer on startup
    - Initializes the drift detector with training data
    - Cleans up on shutdown (if needed)
    """
    global model, vectorizer, drift_detector
    
    # Startup: Load model and initialize drift detector
    print("=" * 60)
    print("FastAPI Application Startup")
    print("=" * 60)
    
    try:
        print("\n[1] Loading production model from MLflow...")
        model, vectorizer = load_production_model()
        print("✓ Model loaded successfully!")
        
        print("\n[2] Initializing drift detector...")
        # Load training data to initialize drift detector
        try:
            df = load_support_ticket_data()
            train_df, _, _ = train_val_test_split(df)
            train_texts = preprocess_texts(train_df['text'].tolist())
            
            # Compute reference lengths (number of words per ticket)
            reference_lengths = [len(text.split()) for text in train_texts]
            
            # Initialize drift detector
            drift_detector = DriftDetector(reference_lengths, window_size=100)
            print("✓ Drift detector initialized!")
            
        except Exception as e:
            print(f"⚠ Warning: Could not initialize drift detector: {e}")
            print("  Drift detection will be disabled.")
            drift_detector = None
        
        print("\n" + "=" * 60)
        print("Application ready to serve requests!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to load model: {e}")
        print("  Make sure you have trained a model first:")
        print("    python -m src.train")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (if needed)
    print("\nShutting down application...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Support Ticket Classifier API",
    description="MLOps API for classifying customer support tickets into categories",
    version="1.0.0",
    lifespan=lifespan
)

# Add Prometheus instrumentation
# This automatically tracks request count, latency, and error rate
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status "ok" if the service is healthy.
    """
    return HealthResponse(status="ok")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        Dictionary with model version, stage, and other metadata.
    """
    try:
        info = get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict categories for a batch of support tickets.
    
    This endpoint:
    1. Takes a list of support ticket texts
    2. Preprocesses and vectorizes them
    3. Makes predictions using the loaded model
    4. Updates drift detection statistics
    5. Returns predictions with optional confidence scores
    
    Args:
        request: PredictionRequest containing list of ticket texts.
    
    Returns:
        PredictionResponse with predictions for each ticket.
    
    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    global model, vectorizer, drift_detector
    
    # Check if model is loaded
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Make predictions
        predictions, probabilities = predict_tickets(
            request.tickets,
            return_proba=True
        )
        
        # Build response with predictions and confidence scores
        results = []
        for i, (ticket, pred_label) in enumerate(zip(request.tickets, predictions)):
            confidence = None
            if probabilities is not None:
                # Get the confidence for the predicted class
                # Find the index of the predicted label in the model's classes
                label_index = list(model.classes_).index(pred_label)
                confidence = float(probabilities[i][label_index])
            
            results.append(TicketPrediction(
                text=ticket,
                predicted_label=pred_label,
                confidence=confidence
            ))
        
        # Update drift detector with new tickets
        if drift_detector is not None:
            drift_detector.update(request.tickets)
            drift_score = drift_detector.compute_drift_score()
            
            # Update Prometheus metric
            model_info = get_model_info()
            model_version = model_info.get("version", "unknown")
            drift_score_metric.labels(model_version=model_version).set(drift_score)
        
        return PredictionResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/drift/stats", tags=["Drift Detection"])
async def drift_stats():
    """
    Get current drift detection statistics.
    
    Returns:
        Dictionary with drift detection statistics and current drift score.
    """
    global drift_detector
    
    if drift_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized."
        )
    
    stats = drift_detector.get_statistics()
    return stats

