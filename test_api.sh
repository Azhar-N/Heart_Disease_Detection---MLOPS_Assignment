#!/bin/bash
# Script to test the API endpoints

API_URL="http://localhost:8000"

echo "Testing Heart Disease Prediction API"
echo "===================================="

# Health check
echo -e "\n1. Health Check:"
curl -s "$API_URL/health" | python -m json.tool

# Prediction
echo -e "\n2. Prediction:"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json | python -m json.tool

# Metrics
echo -e "\n3. Metrics:"
curl -s "$API_URL/metrics" | head -20

echo -e "\n\nTesting completed!"
