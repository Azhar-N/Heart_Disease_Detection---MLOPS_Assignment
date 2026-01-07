#!/bin/bash
# Script to run model training

echo "Downloading dataset..."
python src/data/download.py

echo "Training models..."
python src/models/train.py

echo "Training completed!"
