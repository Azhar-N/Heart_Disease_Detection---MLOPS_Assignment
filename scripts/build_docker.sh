#!/bin/bash
# Script to build Docker image

echo "Building Docker image..."
docker build -t heart-disease-api:latest -f docker/Dockerfile .

echo "Docker image built successfully!"
echo "To run: docker run -d -p 8000:8000 heart-disease-api:latest"
