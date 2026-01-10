# Heart Disease Prediction - MLOps Assignment

End-to-End ML Model Development, CI/CD, and Production Deployment

## Project Overview

This project implements a complete MLOps pipeline for predicting heart disease risk based on patient health data. The solution includes:

- Data acquisition and EDA
- Feature engineering and model development (Logistic Regression & Random Forest)
- Experiment tracking with MLflow
- Model packaging and reproducibility
- CI/CD pipeline with GitHub Actions
- Docker containerization
- Kubernetes deployment
- Monitoring and logging

## Dataset

**Title:** Heart Disease UCI Dataset  
**Source:** UCI Machine Learning Repository  
**Features:** 14+ features (age, sex, blood pressure, cholesterol, etc.)  
**Target:** Binary classification (presence/absence of heart disease)

## Project Structure

```
.
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/               # Processed dataset
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # Data download script
│   │   └── preprocessing.py    # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training script
│   │   └── predict.py          # Prediction utilities
│   └── api/
│       ├── __init__.py
│       └── app.py              # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Data processing tests
│   └── test_models.py           # Model tests
├── mlruns/                     # MLflow experiment tracking
├── models/                      # Saved models
├── docker/
│   └── Dockerfile              # Docker configuration
├── k8s/
│   ├── deployment.yaml         # Kubernetes deployment
│   ├── service.yaml            # Kubernetes service
│   └── ingress.yaml            # Kubernetes ingress
├── monitoring/
│   └── prometheus.yml          # Prometheus configuration
├── .github/
│   └── workflows/
│       └── ci_cd.yml           # GitHub Actions workflow
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- Python 3.12+
- Docker (Optional - for containerization)
- Kubernetes (Optional - Minikube/Docker Desktop or cloud provider)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Assignment1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
python src/data/download.py
```

## Usage

### 1. Exploratory Data Analysis

Run the EDA notebook:
```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Train Models

Train models with MLflow tracking:
```bash
python src/models/train.py
```

### 3. Run Tests

Execute unit tests:
```bash
pytest tests/
```

### 4. Run API Locally

Start the FastAPI server:
```bash
python src/api/app.py
```

Or using Docker:
```bash
docker build -t heart-disease-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-disease-api
```

### 5. Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /predict` - Predict heart disease risk

Example prediction request:
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
- Lints code (flake8, black)
- Runs unit tests
- Trains models
- Builds Docker image
- Deploys to staging (if configured)

## Monitoring

- Prometheus metrics available at `/metrics`
- Grafana dashboards (if configured)
- API logs in structured JSON format
-

## License

MIT License

## Author

MLOps Assignment - S1-25_AIMLCZG523
