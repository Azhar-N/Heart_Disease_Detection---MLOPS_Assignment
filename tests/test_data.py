"""
Unit tests for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import load_and_preprocess_data, HeartDiseasePreprocessor


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_load_data_shape(self):
        """Test that data loads with correct shape."""
        X, y, feature_names = load_and_preprocess_data()

        assert X.shape[1] == 13, "Should have 13 features"
        assert len(feature_names) == 13, "Should have 13 feature names"
        assert len(X) == len(y), "X and y should have same length"
        assert len(y) > 0, "Should have data points"

    def test_load_data_types(self):
        """Test data types."""
        X, y, feature_names = load_and_preprocess_data()

        assert isinstance(X, pd.DataFrame), "X should be DataFrame"
        assert isinstance(y, pd.Series), "y should be Series"
        assert isinstance(feature_names, list), "feature_names should be list"

    def test_target_binary(self):
        """Test that target is binary."""
        X, y, feature_names = load_and_preprocess_data()

        unique_values = y.unique()
        assert set(unique_values).issubset({0, 1}), "Target should be binary"

    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        X, y, feature_names = load_and_preprocess_data()

        preprocessor = HeartDiseasePreprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0], "Should preserve rows"
        assert X_transformed.shape[1] == X.shape[1], "Should preserve columns"
        assert isinstance(X_transformed, np.ndarray), "Should return numpy array"

    def test_preprocessor_transform_without_fit(self):
        """Test that transform fails without fit."""
        X, y, feature_names = load_and_preprocess_data()

        preprocessor = HeartDiseasePreprocessor()

        with pytest.raises(ValueError):
            preprocessor.transform(X)

    def test_preprocessor_save_load(self):
        """Test preprocessor save and load."""
        X, y, feature_names = load_and_preprocess_data()

        preprocessor = HeartDiseasePreprocessor()
        preprocessor.fit(X)

        # Save
        save_path = "test_preprocessor.joblib"
        preprocessor.save(save_path)

        # Load
        loaded_preprocessor = HeartDiseasePreprocessor.load(save_path)

        # Test transform
        original_output = preprocessor.transform(X)
        loaded_output = loaded_preprocessor.transform(X)

        np.testing.assert_array_almost_equal(original_output, loaded_output)

        # Cleanup
        Path(save_path).unlink()

    def test_no_missing_values_after_preprocessing(self):
        """Test that preprocessing handles missing values."""
        X, y, feature_names = load_and_preprocess_data()

        preprocessor = HeartDiseasePreprocessor()
        X_transformed = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any(), "Should not have NaN values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
