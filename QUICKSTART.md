# Quick Start Guide

## Prerequisites

- Python 3.12+
- Docker (optional, for containerization)
- Kubernetes (optional, for deployment)

## Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Dataset

```bash
python src/data/download.py
```

This will download the Heart Disease dataset from UCI ML Repository to `data/raw/heart_disease.csv`.

## Step 3: Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

Execute all cells in the notebook to generate EDA visualizations.

## Step 4: Train Models

```bash
python src/models/train.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression and Random Forest models
- Log experiments to MLflow
- Save models to `models/` directory

## Step 5: View MLflow Experiments

```bash
mlflow ui
```

Open http://localhost:5000 in your browser to view experiment tracking.

**What to Check:**
- Experiment runs for Logistic Regression and Random Forest
- Parameters logged (hyperparameters, random seeds)
- Metrics logged (accuracy, precision, recall, ROC-AUC, CV scores)
- Model artifacts (saved models, preprocessors)
- Feature importance plots (for Random Forest)

**Experiment Tracking Features:**
- All model parameters logged
- Comprehensive metrics tracking
- Model versioning
- Artifact storage
- Experiment comparison

## Step 6: Run Tests

```bash
pytest tests/ -v
```

This runs unit tests for:
- Data preprocessing pipeline
- Model prediction utilities
- Code coverage reporting

## Step 6b: Verify Model Packaging

After training, verify that models and preprocessors are saved:

```bash
# Check models directory
ls models/

# Should contain:
# - best_model.joblib
# - logistic_regression.joblib
# - random_forest.joblib
# - preprocessor.joblib
```

**Model Packaging Features:**
- Models saved in Joblib format for reproducibility
- Preprocessor pipeline saved separately
- All dependencies documented in `requirements.txt`
- Fixed random seeds ensure reproducibility

## Step 7: Run API Locally

```bash
python src/api/app.py
```

Or using uvicorn:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Step 8: Test API

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Verify Monitoring & Logging

**Check Prometheus Metrics:**
```bash
curl http://localhost:8000/metrics
```

**View API Logs:**
The API logs all requests in structured JSON format. Check the console output for:
- Request timestamps
- Prediction results
- Latency metrics
- Error logs (if any)

**Monitoring Features:**
- Prometheus metrics endpoint (`/metrics`)
- Structured JSON logging
- Request/response tracking
- Performance metrics (latency, prediction counts)

## Step 9: Build Docker Image (Optional - Requires Docker Desktop)

**Note:** Docker is optional for local testing. You can skip this step if you don't have Docker installed.

### Install Docker Desktop (Windows)

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install and restart your computer if prompted
3. Start Docker Desktop and wait for it to be ready

### Build Docker Image

```bash
docker build -t heart-disease-api:latest -f docker/Dockerfile .
```

## Step 10: Run Docker Container (Optional)

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

**Note:** If Docker is not installed, you can test the API locally using Step 7 (Run API Locally) instead.

## Step 11: Deploy to Kubernetes (Optional)

**Prerequisites:** Kubernetes cluster must be running (Minikube, Docker Desktop Kubernetes, or cloud provider)

### Option A: Using Minikube

```bash
# Start Minikube
minikube start

# Load Docker image into Minikube
minikube image load heart-disease-api:latest

# Apply deployment
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# Port forward to access locally
kubectl port-forward service/heart-disease-api-service 8000:80
```

### Option B: Using Docker Desktop Kubernetes

```bash
# Enable Kubernetes in Docker Desktop Settings
# Then apply manifests:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Port forward
kubectl port-forward service/heart-disease-api-service 8000:80
```

### Option C: Skip Validation (if Kubernetes not available)

If you get validation errors, you can skip validation:

```bash
kubectl apply -f k8s/deployment.yaml --validate=false
kubectl apply -f k8s/service.yaml --validate=false
```

**Note:** For assignment purposes, you can document that Kubernetes manifests are provided and ready for deployment. Local testing requires a Kubernetes cluster.

## Troubleshooting

### Issue: Module not found
**Solution:** Make sure you're in the project root and virtual environment is activated.

### Issue: Dataset download fails
**Solution:** Check internet connection. You can manually download from:
https://archive.ics.uci.edu/ml/datasets/heart+disease

Save as `data/raw/heart_disease.csv` with columns:
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

### Issue: Models not found when running API
**Solution:** Train models first using `python src/models/train.py`

### Issue: Port already in use
**Solution:** Change port in `src/api/app.py` or use:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

### Issue: Docker command not recognized
**Solution:** Docker is optional for local testing. You can:
1. **Skip Docker:** Test the API locally using Step 7 (Run API Locally)
2. **Install Docker Desktop:** Download from https://www.docker.com/products/docker-desktop/
   - After installation, restart your computer
   - Start Docker Desktop and wait for it to be ready
   - Then try the Docker commands again

## Step 12: Verify CI/CD Pipeline (GitHub Actions)

If you've pushed to GitHub, check the Actions tab:

1. **View Workflow Runs:**
   - Go to your repository on GitHub
   - Click on "Actions" tab
   - View workflow runs for linting, testing, training, and Docker build

2. **What CI/CD Does:**
   - ✅ Code linting (flake8, black)
   - ✅ Unit tests with coverage
   - ✅ Model training
   - ✅ Docker image building
   - ✅ Artifact storage

3. **For Local Testing:**
   ```bash
   # Run linting
   flake8 src/ tests/
   black --check src/ tests/
   
   # Run tests with coverage
   pytest tests/ -v --cov=src --cov-report=html
   ```

## Assignment Checklist

Use this checklist to ensure all requirements are met:

- [ ] **Data Acquisition & EDA** - Dataset downloaded, EDA notebook executed
- [ ] **Feature Engineering & Model Development** - Models trained (LR & RF)
- [ ] **Experiment Tracking** - MLflow UI shows experiments with metrics
- [ ] **Model Packaging** - Models saved in `models/` directory
- [ ] **CI/CD Pipeline** - GitHub Actions workflow runs successfully
- [ ] **Containerization** - Docker image builds successfully
- [ ] **Production Deployment** - Kubernetes manifests ready (or documented)
- [ ] **Monitoring & Logging** - `/metrics` endpoint works, logs visible
- [ ] **Documentation** - All documentation files reviewed

## Next Steps

1. Review the EDA notebook for data insights
2. Check MLflow UI for experiment tracking
3. Review `ARCHITECTURE.md` for system design
4. Review `DEPLOYMENT.md` for deployment details