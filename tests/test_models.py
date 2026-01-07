"""
Unit tests for model training and prediction.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import load_and_preprocess_data, HeartDiseasePreprocessor
from src.models.predict import HeartDiseasePredictor


class TestModelPrediction:
    """Test model prediction functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y, feature_names = load_and_preprocess_data()
        preprocessor = HeartDiseasePreprocessor()
        X_scaled = preprocessor.fit_transform(X[:10])
        return X_scaled, y[:10]
    
    def test_predictor_initialization(self):
        """Test predictor can be initialized if model exists."""
        # This test will pass if models are trained first
        model_path = Path("models/best_model.joblib")
        preprocessor_path = Path("models/preprocessor.joblib")
        
        if model_path.exists() and preprocessor_path.exists():
            predictor = HeartDiseasePredictor()
            assert predictor.model is not None
            assert predictor.preprocessor is not None
    
    def test_prediction_output_format(self):
        """Test prediction output format."""
        model_path = Path("models/best_model.joblib")
        preprocessor_path = Path("models/preprocessor.joblib")
        
        if model_path.exists() and preprocessor_path.exists():
            predictor = HeartDiseasePredictor()
            
            # Sample input
            sample_features = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
            
            prediction, probability = predictor.predict(sample_features)
            
            assert prediction in [0, 1], "Prediction should be 0 or 1"
            assert 0 <= probability <= 1, "Probability should be between 0 and 1"
            assert isinstance(prediction, (int, np.integer)), "Prediction should be integer"
            assert isinstance(probability, (float, np.floating)), "Probability should be float"
    
    def test_prediction_with_valid_input(self):
        """Test prediction with valid input shape."""
        model_path = Path("models/best_model.joblib")
        preprocessor_path = Path("models/preprocessor.joblib")
        
        if model_path.exists() and preprocessor_path.exists():
            predictor = HeartDiseasePredictor()
            
            # Valid input: 13 features
            valid_input = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
            
            prediction, probability = predictor.predict(valid_input)
            
            assert prediction is not None
            assert probability is not None


class TestModelTraining:
    """Test model training components."""
    
    def test_data_split(self):
        """Test that data can be split properly."""
        X, y, feature_names = load_and_preprocess_data()
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
