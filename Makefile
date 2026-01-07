.PHONY: help install download train test api docker-build docker-run deploy clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make download     - Download dataset"
	@echo "  make train        - Train models"
	@echo "  make test         - Run tests"
	@echo "  make api          - Run API locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make deploy       - Deploy to Kubernetes"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt

download:
	python src/data/download.py

train:
	python src/models/train.py

test:
	pytest tests/ -v --cov=src --cov-report=html

api:
	python src/api/app.py

docker-build:
	docker build -t heart-disease-api:latest -f docker/Dockerfile .

docker-run:
	docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

deploy:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
