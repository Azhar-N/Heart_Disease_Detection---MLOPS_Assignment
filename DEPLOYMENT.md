# Deployment Guide

## Prerequisites

- Docker installed and running
- Kubernetes cluster (Minikube, Docker Desktop, or cloud provider)
- kubectl configured
- Python 3.12+ (for local development)

## Local Development

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python src/data/download.py
```

### 3. Train Models

```bash
python src/models/train.py
```

### 4. Run API Locally

```bash
python src/api/app.py
```

Or using uvicorn:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
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

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t heart-disease-api:latest -f docker/Dockerfile .
```

### 2. Run Container

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

### 3. Verify Deployment

```bash
curl http://localhost:8000/health
```

### 4. Stop Container

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

## Kubernetes Deployment

### Option 1: Minikube (Local)

#### 1. Start Minikube

```bash
minikube start
```

#### 2. Load Docker Image

```bash
# Build image
docker build -t heart-disease-api:latest -f docker/Dockerfile .

# Load into Minikube
minikube image load heart-disease-api:latest
```

#### 3. Deploy to Kubernetes

```bash
# Apply deployment and service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services
```

#### 4. Access Service

```bash
# Get service URL
minikube service heart-disease-api-service --url

# Or port forward
kubectl port-forward service/heart-disease-api-service 8000:80
```

### Option 2: Docker Desktop Kubernetes

#### 1. Enable Kubernetes in Docker Desktop

- Open Docker Desktop
- Go to Settings → Kubernetes
- Enable Kubernetes

#### 2. Deploy

```bash
# Build and tag image
docker build -t heart-disease-api:latest -f docker/Dockerfile .

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

#### 3. Access Service

```bash
# Port forward
kubectl port-forward service/heart-disease-api-service 8000:80
```

### Option 3: Cloud Provider (GKE/EKS/AKS)

#### 1. Build and Push Image

```bash
# Tag for your registry
docker tag heart-disease-api:latest gcr.io/PROJECT_ID/heart-disease-api:latest

# Push to registry
docker push gcr.io/PROJECT_ID/heart-disease-api:latest
```

#### 2. Update Deployment

Edit `k8s/deployment.yaml` to use your image:
```yaml
image: gcr.io/PROJECT_ID/heart-disease-api:latest
imagePullPolicy: Always
```

#### 3. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

#### 4. Get External IP

```bash
kubectl get service heart-disease-api-service
```

## Monitoring Setup

### Prometheus

#### 1. Install Prometheus

```bash
# Using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack
```

#### 2. Configure Scraping

Update `monitoring/prometheus.yml` with your service endpoints.

#### 3. Access Grafana

```bash
kubectl port-forward service/prometheus-grafana 3000:80
```

Default credentials:
- Username: admin
- Password: prom-operator

## Troubleshooting

### Check Pod Logs

```bash
kubectl logs -l app=heart-disease-api
```

### Check Pod Status

```bash
kubectl describe pod <pod-name>
```

### Check Service

```bash
kubectl describe service heart-disease-api-service
```

### Restart Deployment

```bash
kubectl rollout restart deployment/heart-disease-api
```

## Verification Checklist

- [ ] API responds to `/health` endpoint
- [ ] `/predict` endpoint returns valid predictions
- [ ] `/metrics` endpoint exposes Prometheus metrics
- [ ] Pods are running and healthy
- [ ] Service is accessible
- [ ] Logs show no errors
- [ ] Metrics are being collected

## Production Considerations

1. **Secrets Management**: Use Kubernetes secrets for sensitive data
2. **Resource Limits**: Adjust based on load testing
3. **Auto-scaling**: Configure HPA for automatic scaling
4. **Ingress**: Set up proper ingress controller for external access
5. **SSL/TLS**: Configure certificates for HTTPS
6. **Backup**: Regular backups of MLflow artifacts and models
7. **Monitoring**: Set up alerts for critical metrics
