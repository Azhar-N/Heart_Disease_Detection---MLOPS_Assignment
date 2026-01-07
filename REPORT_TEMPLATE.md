# MLOps Assignment Report
## Heart Disease Prediction - End-to-End ML Pipeline

**Course:** MLOps (S1-25_AIMLCZG523)  
**Assignment:** Assignment I  
**Student:** [Your Name]  
**Date:** [Date]

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Data Acquisition & Exploratory Data Analysis](#data-acquisition--exploratory-data-analysis)
4. [Feature Engineering & Model Development](#feature-engineering--model-development)
5. [Experiment Tracking](#experiment-tracking)
6. [Model Packaging & Reproducibility](#model-packaging--reproducibility)
7. [CI/CD Pipeline & Automated Testing](#cicd-pipeline--automated-testing)
8. [Model Containerization](#model-containerization)
9. [Production Deployment](#production-deployment)
10. [Monitoring & Logging](#monitoring--logging)
11. [Architecture Overview](#architecture-overview)
12. [Results & Discussion](#results--discussion)
13. [Conclusion](#conclusion)
14. [References](#references)

---

## Executive Summary

This report documents the development and deployment of an end-to-end machine learning pipeline for predicting heart disease risk. The solution implements modern MLOps practices including automated data processing, model training with experiment tracking, CI/CD pipelines, containerization, and cloud deployment with monitoring capabilities.

**Key Achievements:**
- Successfully implemented automated data acquisition and preprocessing
- Trained and evaluated two classification models (Logistic Regression and Random Forest)
- Integrated MLflow for comprehensive experiment tracking
- Established CI/CD pipeline with automated testing
- Containerized the application using Docker
- Deployed to Kubernetes with monitoring integration
- Achieved production-ready deployment with proper logging and metrics

---

## Introduction

### Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and risk assessment can significantly improve patient outcomes. This project aims to build a machine learning classifier to predict the risk of heart disease based on patient health data, deployed as a scalable, monitored API service.

### Objectives

1. Develop a robust ML pipeline for heart disease prediction
2. Implement MLOps best practices for reproducibility and scalability
3. Deploy the solution as a production-ready API service
4. Establish monitoring and logging for operational visibility

### Dataset

**Source:** UCI Machine Learning Repository - Heart Disease Dataset  
**Features:** 13 features including age, sex, blood pressure, cholesterol, etc.  
**Target:** Binary classification (presence/absence of heart disease)  
**Samples:** 303 records

---

## Data Acquisition & Exploratory Data Analysis

### Data Acquisition

The dataset was automatically downloaded from the UCI ML Repository using a Python script (`src/data/download.py`). The script handles:
- Automated download from UCI repository
- Data cleaning and preprocessing
- Target variable conversion to binary format

**Screenshot:** [Include screenshot of data download script execution]

### Exploratory Data Analysis

Comprehensive EDA was performed using Jupyter notebooks (`notebooks/eda.ipynb`). Key findings:

#### Dataset Overview
- **Total Samples:** 303
- **Features:** 13 numeric features
- **Missing Values:** Minimal, handled during preprocessing
- **Class Distribution:** Relatively balanced (54% no disease, 46% disease)

#### Key Visualizations

1. **Target Distribution:** [Include histogram showing class balance]
2. **Feature Distributions:** [Include histograms for key features]
3. **Correlation Heatmap:** [Include correlation matrix visualization]
4. **Feature-Target Relationships:** [Include box plots]

#### Insights
- Age, cholesterol, and maximum heart rate show significant variation
- Some features exhibit correlation with the target variable
- No severe class imbalance requiring special handling

**Screenshots:** [Include EDA visualizations]

---

## Feature Engineering & Model Development

### Feature Engineering

The preprocessing pipeline (`src/data/preprocessing.py`) includes:

1. **Missing Value Handling:** Median imputation for numeric features
2. **Feature Scaling:** StandardScaler for normalization
3. **Reproducibility:** Saved preprocessor for consistent transformations

### Model Development

Two classification models were developed:

#### 1. Logistic Regression
- **Hyperparameters:**
  - C: 1.0
  - Solver: lbfgs
  - Max iterations: 1000
- **Performance:**
  - Accuracy: [Value]
  - Precision: [Value]
  - Recall: [Value]
  - ROC-AUC: [Value]

#### 2. Random Forest
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
- **Performance:**
  - Accuracy: [Value]
  - Precision: [Value]
  - Recall: [Value]
  - ROC-AUC: [Value]

### Model Selection

[Describe model selection process and rationale]

**Best Model:** [Logistic Regression / Random Forest]  
**Selection Criteria:** ROC-AUC score

### Evaluation Metrics

Models were evaluated using:
- **Cross-validation:** 5-fold stratified CV
- **Metrics:** Accuracy, Precision, Recall, ROC-AUC
- **Classification Report:** [Include report]

**Screenshots:** [Include model training outputs and evaluation metrics]

---

## Experiment Tracking

### MLflow Integration

MLflow was integrated for comprehensive experiment tracking:

#### Tracked Components:
1. **Parameters:** Model hyperparameters
2. **Metrics:** Accuracy, Precision, Recall, ROC-AUC, CV scores
3. **Artifacts:** Trained models, preprocessors, feature importance
4. **Metadata:** Run names, timestamps, experiment tags

#### MLflow UI Screenshots:
- [Experiment overview]
- [Run comparison]
- [Model artifacts]
- [Metrics visualization]

### Experiment Summary

- **Total Runs:** [Number]
- **Best Run:** [Run details]
- **Key Findings:** [Insights from experiments]

**Screenshots:** [Include MLflow UI screenshots]

---

## Model Packaging & Reproducibility

### Model Serialization

Models are saved in multiple formats:
- **Joblib:** Primary format for scikit-learn models
- **MLflow:** Integrated model registry
- **Preprocessor:** Separate serialization for reproducibility

### Requirements Management

**requirements.txt** includes:
- Core ML libraries (scikit-learn, pandas, numpy)
- MLflow for tracking
- FastAPI for serving
- Testing frameworks (pytest)
- Code quality tools (flake8, black)

### Reproducibility Features

1. **Fixed Random Seeds:** All random operations use seed=42
2. **Versioned Dependencies:** Pinned package versions
3. **Preprocessing Pipeline:** Saved and versioned
4. **Model Artifacts:** Stored with metadata

**File Structure:**
```
models/
├── best_model.joblib
├── logistic_regression.joblib
├── random_forest.joblib
└── preprocessor.joblib
```

---

## CI/CD Pipeline & Automated Testing

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci_cd.yml`) includes:

#### Stages:
1. **Linting:** flake8 and black code quality checks
2. **Testing:** pytest with coverage reporting
3. **Training:** Model training with MLflow
4. **Building:** Docker image construction

### Unit Tests

Test suite (`tests/`) covers:
- **Data Processing:** Preprocessing pipeline tests
- **Model Prediction:** Prediction format and validation tests
- **API Endpoints:** (if applicable)

**Test Coverage:** [Percentage]

**Screenshots:** [Include GitHub Actions workflow runs]

### Pipeline Artifacts

- MLflow runs
- Trained models
- Docker images
- Test reports

---

## Model Containerization

### Docker Configuration

**Dockerfile** (`docker/Dockerfile`) features:
- Multi-stage build for optimization
- Python 3.8 base image
- Minimal dependencies
- Health check configuration
- Proper port exposure

### Container Build

```bash
docker build -t heart-disease-api:latest -f docker/Dockerfile .
```

### Container Testing

```bash
docker run -d -p 8000:8000 heart-disease-api:latest
curl http://localhost:8000/health
```

**Screenshots:** [Include Docker build and test outputs]

### Image Size Optimization

- **Base Image:** python:3.8-slim
- **Multi-stage Build:** Reduced final image size
- **Final Size:** [Size in MB]

---

## Production Deployment

### Kubernetes Deployment

#### Deployment Configuration

**deployment.yaml** includes:
- 3 replicas for high availability
- Resource limits (CPU: 500m, Memory: 512Mi)
- Liveness and readiness probes
- Health check endpoints

#### Service Configuration

**service.yaml** exposes:
- LoadBalancer type for external access
- Port mapping (80 → 8000)

#### Deployment Steps

1. Build Docker image
2. Load into Kubernetes cluster
3. Apply deployment manifests
4. Verify service accessibility

**Screenshots:** [Include Kubernetes deployment screenshots]
- Pod status
- Service details
- Ingress configuration (if applicable)

### Deployment Verification

- [ ] Pods running successfully
- [ ] Service accessible
- [ ] Health endpoint responding
- [ ] Predictions working correctly
- [ ] Metrics endpoint functional

**Access URL:** [Provide URL or access instructions]

---

## Monitoring & Logging

### Logging Implementation

Structured JSON logging implemented:
- Request/response logging
- Error tracking
- Performance metrics
- Timestamp and module information

### Prometheus Metrics

Exposed metrics:
- `heart_disease_predictions_total`: Total prediction count
- `heart_disease_prediction_latency_seconds`: Prediction latency

**Metrics Endpoint:** `/metrics`

### Monitoring Dashboard

[If Grafana configured, include dashboard screenshots]

**Screenshots:** [Include monitoring dashboards]

---

## Architecture Overview

### System Architecture

[Include architecture diagram from ARCHITECTURE.md]

### Component Interactions

1. **Data Flow:** UCI Repository → Download → Preprocess → Train → Serve
2. **API Flow:** Request → Validate → Preprocess → Predict → Response
3. **Monitoring Flow:** Metrics → Prometheus → Grafana (optional)

### Scalability

- Horizontal scaling via Kubernetes replicas
- Resource limits for stability
- Load balancing for distribution

---

## Results & Discussion

### Model Performance

[Summarize model performance metrics]

### Deployment Success

- Successfully deployed to [Kubernetes / Cloud Provider]
- API responding correctly
- Monitoring operational

### Challenges & Solutions

1. **Challenge:** [Description]
   **Solution:** [Description]

2. **Challenge:** [Description]
   **Solution:** [Description]

### Future Improvements

1. Model retraining pipeline
2. A/B testing framework
3. Advanced monitoring and alerting
4. Model explainability features

---

## Conclusion

This project successfully demonstrates an end-to-end MLOps pipeline for heart disease prediction. Key achievements include:

- Automated data processing and model training
- Comprehensive experiment tracking with MLflow
- Production-ready deployment with Docker and Kubernetes
- Monitoring and logging for operational visibility

The solution follows MLOps best practices and provides a foundation for scalable ML deployments.

---

## References

1. UCI Machine Learning Repository - Heart Disease Dataset
2. MLflow Documentation
3. FastAPI Documentation
4. Kubernetes Documentation
5. Docker Documentation

---

## Appendix

### A. Repository Link

[GitHub Repository URL]

### B. API Documentation

[Include API endpoint documentation]

### C. Deployment Screenshots

[Include all relevant screenshots]

### D. Video Demonstration

[Link to video demonstration]

---

**End of Report**
