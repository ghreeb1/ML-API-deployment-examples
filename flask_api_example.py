# ==========================================================
# Flask ML Example (Production-Ready Version)
#
# Purpose: Serve a Machine Learning model as an HTTP API
# using Flask with production best practices.
# ==========================================================

from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import logging
import os
from typing import Optional, Dict, Any
from functools import wraps
import json

# ----------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("flask_ml_example")

# ----------------------------------------------------------
# Global Model State
# ----------------------------------------------------------
model: Optional[object] = None
MODEL_VERSION = "1.0.0"


# ----------------------------------------------------------
# Model Loading
# ----------------------------------------------------------
def load_model() -> bool:
    """
    Load the ML model into memory.
    
    Uses MODEL_PATH environment variable if present.
    Designed to be idempotent - safe to call multiple times.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model
    
    if model is not None:
        logger.info("Model already loaded")
        return True
    
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        return True
    except FileNotFoundError:
        logger.error(f"✗ Model file not found: {model_path}")
        model = None
        return False
    except Exception as exc:
        logger.exception(f"✗ Failed to load model: {exc}")
        model = None
        return False


# ----------------------------------------------------------
# Create Flask Application
# ----------------------------------------------------------
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size


# ----------------------------------------------------------
# Decorators
# ----------------------------------------------------------
def require_json(f):
    """Decorator to ensure request contains valid JSON."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 415
        return f(*args, **kwargs)
    return decorated_function


def require_model(f):
    """Decorator to ensure model is loaded before processing request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if model is None:
            logger.warning("Request received but model not loaded")
            return jsonify({
                "error": "Model not available",
                "details": "Service not ready. Check /health endpoint."
            }), 503
        return f(*args, **kwargs)
    return decorated_function


# ----------------------------------------------------------
# Error Handlers
# ----------------------------------------------------------
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({
        "error": "Bad Request",
        "message": str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 Request Entity Too Large errors."""
    return jsonify({
        "error": "Request Too Large",
        "message": "Request payload exceeds maximum allowed size"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    logger.exception("Internal server error")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


# ----------------------------------------------------------
# Request Validation
# ----------------------------------------------------------
def validate_features(features: Any) -> tuple[bool, Optional[str]]:
    """
    Validate feature input.
    
    Args:
        features: Input to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if features is None:
        return False, "Missing 'features' field in request body"
    
    if not isinstance(features, (list, tuple)):
        return False, "'features' must be a list or array"
    
    if len(features) == 0:
        return False, "'features' cannot be empty"
    
    if len(features) > 1000:
        return False, "'features' exceeds maximum length of 1000"
    
    if not all(isinstance(x, (int, float)) for x in features):
        return False, "All features must be numeric"
    
    if not all(np.isfinite(x) for x in features):
        return False, "All features must be finite (no NaN or Infinity)"
    
    return True, None


# ----------------------------------------------------------
# API Routes
# ----------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    """
    Root endpoint - API information.
    
    Returns basic API metadata and available endpoints.
    """
    return jsonify({
        "service": "Flask ML API",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "ready": "/ready"
        }
    })


@app.route("/predict", methods=["POST"])
@require_json
@require_model
def predict():
    """
    Prediction endpoint.
    
    Accepts JSON body with 'features' array and returns model prediction.
    
    Request Body:
        {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    
    Response (200):
        {
            "prediction": [0],
            "model_version": "1.0.0"
        }
    
    Error Responses:
        400 - Invalid input
        415 - Invalid content type
        503 - Model not available
    """
    try:
        data = request.get_json()
        features = data.get("features")
        
        # Validate input
        is_valid, error_msg = validate_features(features)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        # Prepare input array
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)
        
        logger.info(f"Prediction successful - Input shape: {features_array.shape}")
        
        return jsonify({
            "prediction": prediction.tolist(),
            "model_version": MODEL_VERSION
        }), 200
        
    except ValueError as exc:
        logger.error(f"Invalid input shape: {exc}")
        return jsonify({
            "error": "Invalid input shape or format",
            "details": str(exc)
        }), 400
    except Exception as exc:
        logger.exception(f"Prediction error: {exc}")
        return jsonify({
            "error": "Prediction failed",
            "details": "Internal error during prediction"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Used by orchestrators and load balancers to determine service health.
    Always returns 200, but indicates if model is loaded.
    
    Response:
        {
            "status": "healthy" | "degraded",
            "model_loaded": true | false,
            "model_path": "model.pkl",
            "version": "1.0.0"
        }
    """
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": model_path,
        "version": MODEL_VERSION
    }), 200


@app.route("/ready", methods=["GET"])
def ready():
    """
    Readiness probe endpoint.
    
    Returns 200 only if model is loaded and service is ready.
    Use this for Kubernetes readiness probes.
    
    Response (200):
        {"status": "ready"}
    
    Response (503):
        {"status": "not ready", "reason": "model not loaded"}
    """
    if model is None:
        return jsonify({
            "status": "not ready",
            "reason": "model not loaded"
        }), 503
    
    return jsonify({"status": "ready"}), 200


# ----------------------------------------------------------
# Application Startup
# ----------------------------------------------------------
def initialize_app():
    """Initialize application by loading model."""
    logger.info("Initializing Flask ML API...")
    success = load_model()
    if success:
        logger.info("Application initialization complete")
    else:
        logger.warning("Application started but model failed to load")


# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------
if __name__ == "__main__":
    # Load model at startup for development
    initialize_app()
    
    # Run development server
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    )

# ----------------------------------------------------------
# Production Deployment Instructions:
#
# Using Gunicorn (recommended):
#   gunicorn flask_api_example:app -w 4 -b 0.0.0.0:5000 --preload
#
# Using Waitress:
#   waitress-serve --port=5000 flask_api_example:app
#
# Docker CMD example:
#   CMD ["gunicorn", "flask_api_example:app", "-w", "4", "-b", "0.0.0.0:5000"]
#
# For production, always use a proper WSGI server (Gunicorn, uWSGI, Waitress)
# instead of the built-in Flask development server.
# ----------------------------------------------------------
