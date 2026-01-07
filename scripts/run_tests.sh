#!/bin/bash
# Script to run tests

echo "Running unit tests..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo "Tests completed!"
