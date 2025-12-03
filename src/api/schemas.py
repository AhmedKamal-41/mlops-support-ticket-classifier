"""
Pydantic schemas for FastAPI request/response models.

This module defines the data structures used for API requests and responses,
ensuring type safety and automatic validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for ticket classification predictions.
    
    Attributes:
        tickets: List of support ticket text strings to classify.
    """
    tickets: List[str] = Field(
        ...,
        description="List of support ticket texts to classify",
        min_items=1,
        max_items=100  # Limit batch size for performance
    )
    
    class Config:
        schema_extra = {
            "example": {
                "tickets": [
                    "I was double charged for my subscription",
                    "The app keeps crashing when I open it",
                    "I can't log into my account"
                ]
            }
        }


class TicketPrediction(BaseModel):
    """
    Prediction result for a single support ticket.
    
    Attributes:
        text: The original ticket text.
        predicted_label: The predicted category label.
        confidence: Optional confidence score (probability) for the prediction.
    """
    text: str = Field(..., description="Original ticket text")
    predicted_label: str = Field(..., description="Predicted category label")
    confidence: Optional[float] = Field(
        None,
        description="Confidence score (probability) for the prediction",
        ge=0.0,
        le=1.0
    )


class PredictionResponse(BaseModel):
    """
    Response model containing predictions for all tickets.
    
    Attributes:
        results: List of prediction results, one for each input ticket.
    """
    results: List[TicketPrediction] = Field(
        ...,
        description="List of predictions, one for each input ticket"
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status: Health status (typically "ok").
    """
    status: str = Field(default="ok", description="Service health status")

