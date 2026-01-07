# System Architecture

## Overview

This document describes the architecture of the Heart Disease Prediction MLOps system.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
│                    (UCI ML Repository)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Acquisition Layer                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  download.py - Dataset download script                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  preprocessing.py - Data cleaning & transformation      │   │
│  │  EDA Notebook - Exploratory Data Analysis               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training Layer                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  train.py - Model training with MLflow tracking         │   │
│  │  • Logistic Regression                                   │   │
│  │  • Random Forest                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MLflow - Experiment Tracking                            │   │
│  │  • Parameters, Metrics, Artifacts                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Serving Layer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (app.py)                            │   │
│  │  • /predict endpoint                                     │   │
│  │  • /health endpoint                                      │   │
│  │  • /metrics endpoint (Prometheus)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Containerization Layer                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Docker Container                                        │   │
│  │  • Multi-stage build                                     │   │
│  │  • Optimized image size                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Kubernetes Deployment                                   │   │
│  │  • Deployment (3 replicas)                               │   │
│  │  • Service (LoadBalancer)                                │   │
│  │  • Ingress (optional)                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Monitoring Layer                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Prometheus - Metrics Collection                         │   │
│  │  Grafana - Visualization (optional)                      │   │
│  │  Structured Logging                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## CI/CD Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Repository                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              GitHub Actions Workflow                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Lint (flake8, black)                                 │   │
│  │  2. Test (pytest with coverage)                          │   │
│  │  3. Train (model training with MLflow)                   │   │
│  │  4. Build (Docker image)                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Artifact Storage                              │
│  • MLflow runs                                                  │
│  • Trained models                                               │
│  • Docker images                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer
- **download.py**: Automated dataset acquisition from UCI ML Repository
- **preprocessing.py**: Reusable preprocessing pipeline with scaling and imputation

### 2. Model Layer
- **train.py**: Model training script supporting multiple algorithms
- **MLflow**: Experiment tracking, model versioning, and artifact storage
- **predict.py**: Prediction utilities for inference

### 3. API Layer
- **FastAPI**: Modern, fast web framework for building APIs
- **Endpoints**:
  - `/predict`: Main prediction endpoint
  - `/health`: Health check for Kubernetes probes
  - `/metrics`: Prometheus metrics endpoint

### 4. Infrastructure Layer
- **Docker**: Containerization for consistent deployment
- **Kubernetes**: Orchestration for scalable deployment
- **Monitoring**: Prometheus metrics and structured logging

### 5. CI/CD Layer
- **GitHub Actions**: Automated testing, training, and building
- **Quality Gates**: Linting, testing, and validation before deployment

## Data Flow

1. **Training Phase**:
   - Download dataset → Preprocess → Train models → Log to MLflow → Save models

2. **Inference Phase**:
   - Receive request → Preprocess input → Load model → Predict → Return result

3. **Monitoring Phase**:
   - Collect metrics → Store in Prometheus → Visualize in Grafana (optional)

## Scalability Considerations

- **Horizontal Scaling**: Kubernetes deployment with multiple replicas
- **Resource Limits**: CPU and memory limits defined in deployment
- **Health Checks**: Liveness and readiness probes for reliability
- **Load Balancing**: Service type LoadBalancer for external access

## Security Considerations

- Input validation via Pydantic models
- Resource limits to prevent resource exhaustion
- Health checks to ensure service availability
- Structured logging for audit trails
