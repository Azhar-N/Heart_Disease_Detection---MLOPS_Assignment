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
