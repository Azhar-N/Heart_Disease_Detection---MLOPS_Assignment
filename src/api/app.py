# """
# FastAPI application for heart disease prediction API.
# Includes monitoring endpoints and structured logging.
# """
# import sys
# from pathlib import Path

# # Add src to path
# sys.path.append(str(Path(__file__).parent.parent))

# import logging
# import time
# from datetime import datetime
# from typing import List, Optional
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# from starlette.responses import Response

# from models.predict import HeartDiseasePredictor


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(name)s"}',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# # Prometheus metrics
# PREDICTION_COUNTER = Counter(
#     'heart_disease_predictions_total',
#     'Total number of predictions made',
#     ['prediction']
# )

# PREDICTION_LATENCY = Histogram(
#     'heart_disease_prediction_latency_seconds',
#     'Prediction latency in seconds'
# )

# # Initialize FastAPI app
# app = FastAPI(
#     title="Heart Disease Prediction API",
#     description="MLOps Assignment: Heart Disease Risk Prediction API",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize predictor
# try:
#     predictor = HeartDiseasePredictor()
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {e}")
#     predictor = None


# # Request/Response models
# class HeartDiseaseInput(BaseModel):
#     """Input model for prediction request."""
#     age: int = Field(..., ge=0, le=120, description="Age in years")
#     sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
#     cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
#     trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure")
#     chol: int = Field(..., ge=0, le=600, description="Serum cholesterol")
#     fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
#     restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results")
#     thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
#     exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
#     oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
#     slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
#     ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by flourosopy")
#     thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")


# class PredictionResponse(BaseModel):
#     """Response model for prediction."""
#     prediction: int = Field(..., description="Prediction (0=no disease, 1=disease)")
#     probability: float = Field(..., description="Probability of heart disease")
#     timestamp: str = Field(..., description="Prediction timestamp")


# @app.get("/")
# async def root():
#     """Root endpoint."""
#     return {
#         "message": "Heart Disease Prediction API",
#         "version": "1.0.0",
#         "status": "operational"
#     }


# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     if predictor is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "model_loaded": predictor is not None
#     }


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(input_data: HeartDiseaseInput):
#     """
#     Predict heart disease risk based on patient data.

#     Args:
#         input_data: Patient health data

#     Returns:
#         Prediction result with probability
#     """
#     if predictor is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     start_time = time.time()

#     try:
#         # Convert input to numpy array
#         features = np.array([[
#             input_data.age,
#             input_data.sex,
#             input_data.cp,
#             input_data.trestbps,
#             input_data.chol,
#             input_data.fbs,
#             input_data.restecg,
#             input_data.thalach,
#             input_data.exang,
#             input_data.oldpeak,
#             input_data.slope,
#             input_data.ca,
#             input_data.thal
#         ]])

#         # Make prediction
#         prediction, probability = predictor.predict(features)

#         # Record metrics
#         latency = time.time() - start_time
#         PREDICTION_LATENCY.observe(latency)
#         PREDICTION_COUNTER.labels(prediction=str(prediction)).inc()

#         # Log prediction
#         logger.info(
#             f"Prediction made: prediction={prediction}, "
#             f"probability={probability:.4f}, latency={latency:.4f}s"
#         )

#         return PredictionResponse(
#             prediction=prediction,
#             probability=probability,
#             timestamp=datetime.now().isoformat()
#         )

#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# @app.get("/metrics")
# async def metrics():
#     """Prometheus metrics endpoint."""
#     return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI application for heart disease prediction API.
Includes monitoring endpoints and structured logging.
"""

import logging
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from src.models.predict import HeartDiseasePredictor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"message": "%(message)s", "module": "%(name)s"}'
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["prediction"],
)

PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Prediction latency in seconds",
)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment: Heart Disease Risk Prediction API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = HeartDiseasePredictor()
    logger.info("Model loaded successfully")
except Exception as exc:
    logger.error("Failed to load model: %s", exc)
    predictor = None


class HeartDiseaseInput(BaseModel):
    """Input model for prediction request."""

    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=0, le=300)
    chol: int = Field(..., ge=0, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=0, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: int
    probability: float
    timestamp: str


@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HeartDiseaseInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features = np.array(
            [
                [
                    input_data.age,
                    input_data.sex,
                    input_data.cp,
                    input_data.trestbps,
                    input_data.chol,
                    input_data.fbs,
                    input_data.restecg,
                    input_data.thalach,
                    input_data.exang,
                    input_data.oldpeak,
                    input_data.slope,
                    input_data.ca,
                    input_data.thal,
                ]
            ]
        )

        prediction, probability = predictor.predict(features)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(prediction=str(prediction)).inc()

        logger.info(
            "Prediction made: prediction=%s probability=%.4f latency=%.4fs",
            prediction,
            probability,
            latency,
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
