"""
Prediction utilities for heart disease prediction.
"""

import sys
import joblib
import numpy as np
from pathlib import Path
from src.data.preprocessing import HeartDiseasePreprocessor


def _get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate from src/models/predict.py to project root
    project_root = current_file.parent.parent.parent
    # Ensure src is on sys.path for module imports
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    return project_root


class HeartDiseasePredictor:
    """Wrapper class for making predictions with trained models."""

    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize predictor with model and preprocessor.

        Args:
            model_path: Path to saved model (default: models/best_model.joblib)
            preprocessor_path: Path to saved preprocessor (default: models/preprocessor.joblib)
        """
        project_root = _get_project_root()

        # Resolve model path
        if model_path is None:
            model_path = project_root / "models" / "best_model.joblib"
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = project_root / model_path

        # Resolve preprocessor path
        if preprocessor_path is None:
            preprocessor_path = project_root / "models" / "preprocessor.joblib"
        else:
            preprocessor_path = Path(preprocessor_path)
            if not preprocessor_path.is_absolute():
                preprocessor_path = project_root / preprocessor_path

        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train models first using 'python src/models/train.py'"
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                f"Please train models first using 'python src/models/train.py'"
            )

        self.model = joblib.load(model_path)
        self.preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)

    def predict(self, features):
        """
        Predict heart disease risk.

        Args:
            features: Array-like of features

        Returns:
            prediction: 0 or 1
            probability: Probability of heart disease
        """
        # Preprocess features
        features_scaled = self.preprocessor.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        return int(prediction), float(probability)


if __name__ == "__main__":
    # Test prediction
    predictor = HeartDiseasePredictor()

    # Sample input
    sample_features = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

    prediction, probability = predictor.predict(sample_features)
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")
