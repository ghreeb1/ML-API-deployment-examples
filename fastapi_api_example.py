# ==========================================================
# FastAPI ML Example (Production-Ready Version)
#
# Purpose: Serve a Machine Learning model as an HTTP API
# using FastAPI with production best practices.
# ==========================================================

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import joblib
import numpy as np
import logging
import os
from typing import Optional, List

# ----------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("fastapi_ml_example")

# ----------------------------------------------------------
# Global Model State
# ----------------------------------------------------------
model: Optional[object] = None


# ----------------------------------------------------------
# Lifespan Event Handler (Modern FastAPI Pattern)
# ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    
    This replaces the deprecated @app.on_event pattern.
    """
    global model
    
    # Startup: Load model
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"✗ Model file not found: {model_path}")
        model = None
    except Exception as exc:
        logger.exception(f"✗ Failed to load model: {exc}")
        model = None
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (if needed)
    logger.info("Shutting down application")


# ----------------------------------------------------------
# Define FastAPI Application
# ----------------------------------------------------------
app = FastAPI(
    title="FastAPI ML API",
    description="Production-ready API for serving ML model predictions",
    version="1.0.0",
    lifespan=lifespan
)


# ----------------------------------------------------------
# Request/Response Models
# ----------------------------------------------------------
class PredictionInput(BaseModel):
    """
    Input schema for prediction requests.
    
    Attributes:
        features: List of numeric features matching model's expected input shape.
    """
    features: List[float] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Feature vector for prediction",
        examples=[[5.1, 3.5, 1.4, 0.2]]
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate that all features are finite numbers."""
        if not all(np.isfinite(x) for x in v):
            raise ValueError("All features must be finite numbers (no NaN or Inf)")
        return v


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    prediction: List[float]
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model_path: str


# ----------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """
    Root endpoint - API information.
    
    Returns basic API metadata and available endpoints.
    """
    return {
        "service": "FastAPI ML API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["ML Operations"]
)
def predict(data: PredictionInput):
    """
    Prediction endpoint.
    
    Accepts a feature vector and returns model prediction.
    
    Args:
        data: Input features conforming to PredictionInput schema
        
    Returns:
        PredictionOutput with prediction results
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for prediction errors
    """
    if model is None:
        logger.error("Prediction attempted with unloaded model")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Check server logs and /health endpoint."
        )
    
    try:
        # Prepare input array
        features_array = np.array([data.features])
        
        # Make prediction
        prediction = model.predict(features_array)
        
        logger.info(f"Prediction successful - Input shape: {features_array.shape}")
        
        return PredictionOutput(
            prediction=prediction.tolist()
        )
        
    except ValueError as exc:
        logger.error(f"Invalid input shape: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input shape or format: {str(exc)}"
        )
    except Exception as exc:
        logger.exception(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
def health_check():
    """
    Health check endpoint.
    
    Used by orchestrators (Kubernetes, Docker) and load balancers
    to determine service readiness and liveness.
    
    Returns:
        HealthResponse with current service status
    """
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=model_path
    )


@app.get(
    "/ready",
    tags=["Health"]
)
def readiness_check():
    """
    Readiness probe endpoint.
    
    Returns 200 only if model is loaded and service is ready to serve requests.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready - model not loaded"
        )
    return {"status": "ready"}


# ----------------------------------------------------------
# Run Instructions
# ----------------------------------------------------------
# Development:
#   uvicorn fastapi_api_example:app --reload
#
# Production:
#   uvicorn fastapi_api_example:app --host 0.0.0.0 --port 8000 --workers 4
#
# With Gunicorn:
#   gunicorn fastapi_api_example:app -w 4 -k uvicorn.workers.UvicornWorker
#
# Swagger UI:  http://127.0.0.1:8000/docs
# ReDoc:       http://127.0.0.1:8000/redoc
# ----------------------------------------------------------
