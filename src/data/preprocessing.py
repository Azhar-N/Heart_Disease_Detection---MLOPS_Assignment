"""
Data preprocessing pipeline for heart disease prediction.
Handles missing values, encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class HeartDiseasePreprocessor:
    """
    Preprocessing pipeline for heart disease dataset.
    Handles missing values, feature scaling, and ensures reproducibility.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def fit(self, X):
        """
        Fit the preprocessor on training data.

        Args:
            X: DataFrame with features
        """
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)

        # Fit scaler
        self.scaler.fit(X_imputed)
        self.is_fitted = True

        return self

    def transform(self, X):
        """
        Transform data using fitted preprocessor.

        Args:
            X: DataFrame with features

        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Handle missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def save(self, filepath):
        """Save preprocessor to disk."""
        joblib.dump(
            {
                "scaler": self.scaler,
                "imputer": self.imputer,
                "is_fitted": self.is_fitted,
            },
            filepath,
        )
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load preprocessor from disk."""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data["scaler"]
        preprocessor.imputer = data["imputer"]
        preprocessor.is_fitted = data["is_fitted"]
        return preprocessor


def _get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate from src/data/preprocessing.py to project root
    project_root = current_file.parent.parent.parent
    return project_root


def load_and_preprocess_data(data_path=None):
    """
    Load and preprocess the heart disease dataset.

    Args:
        data_path: Path to the raw dataset (default: data/raw/heart_disease.csv)

    Returns:
        X: Features DataFrame
        y: Target Series
        feature_names: List of feature names
    """
    # Resolve data path relative to project root
    if data_path is None:
        project_root = _get_project_root()
        data_path = project_root / "data" / "raw" / "heart_disease.csv"
    else:
        data_path = Path(data_path)
        if not data_path.is_absolute():
            # If relative path, resolve from project root
            project_root = _get_project_root()
            data_path = project_root / data_path

    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please run 'python src/data/download.py' first to download the dataset."
        )

    # Load data
    df = pd.read_csv(data_path)

    # Separate features and target
    feature_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    X = df[feature_columns].copy()
    y = df["target"].copy()

    # Handle missing values (replace with median for numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    return X, y, feature_columns


if __name__ == "__main__":
    # Test preprocessing
    X, y, feature_names = load_and_preprocess_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nMissing values:\n{X.isna().sum()}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
