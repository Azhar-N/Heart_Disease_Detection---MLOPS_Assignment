"""
Data download script for Heart Disease UCI Dataset.
Downloads the dataset from UCI Machine Learning Repository.
"""
import os
import pandas as pd
import requests
from pathlib import Path


def _get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate from src/data/download.py to project root
    project_root = current_file.parent.parent.parent
    return project_root


def download_heart_disease_dataset():
    """
    Download Heart Disease UCI dataset from UCI ML Repository.
    
    The dataset is available at:
    https://archive.ics.uci.edu/ml/datasets/heart+disease
    """
    # Get project root and create data directories
    project_root = _get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (multiple files available, using the processed Cleveland dataset)
    urls = {
        "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    }
    
    # Column names for the dataset
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    print("Downloading Heart Disease UCI Dataset...")
    
    # Download Cleveland dataset (most commonly used)
    url = urls["cleveland"]
    file_path = raw_data_dir / "heart_disease.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        with open(file_path, "w") as f:
            f.write(response.text)
        
        # Load and clean the data
        df = pd.read_csv(file_path, names=column_names, na_values="?")
        
        # Convert target to binary (0 = no disease, 1 = disease)
        # Original: 0 = no disease, 1-4 = disease
        df["target"] = (df["target"] > 0).astype(int)
        
        # Save cleaned version
        df.to_csv(file_path, index=False)
        
        print(f"✓ Dataset downloaded successfully to {file_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Target distribution:\n{df['target'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/heart+disease")
        print("\nSave as: data/raw/heart_disease.csv")
        raise


if __name__ == "__main__":
    download_heart_disease_dataset()
