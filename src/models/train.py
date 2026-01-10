"""
Model training script with MLflow integration.
Trains Logistic Regression and Random Forest classifiers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

from data.preprocessing import load_and_preprocess_data, HeartDiseasePreprocessor


def _get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate from src/models/train.py to project root
    project_root = current_file.parent.parent.parent
    return project_root


# Set MLflow tracking URI (relative to project root)
project_root = _get_project_root()
mlflow.set_tracking_uri(f"file:{project_root / 'mlruns'}")
mlflow.set_experiment("heart_disease_prediction")


def train_logistic_regression(X_train, y_train, X_test, y_test, preprocessor):
    """Train Logistic Regression model."""
    print("\n" + "=" * 50)
    print("Training Logistic Regression Model")
    print("=" * 50)

    with mlflow.start_run(run_name="logistic_regression"):
        # Model parameters
        params = {"C": 1.0, "max_iter": 1000, "random_state": 42, "solver": "lbfgs"}

        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
            }
        )

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log preprocessor
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        preprocessor_path.parent.mkdir(exist_ok=True)
        preprocessor.save(str(preprocessor_path))
        mlflow.log_artifact(str(preprocessor_path), "preprocessor")

        # Print results
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        # Save model
        model_path = project_root / "models" / "logistic_regression.joblib"
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")

        return model, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }


def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    """Train Random Forest model."""
    print("\n" + "=" * 50)
    print("Training Random Forest Model")
    print("=" * 50)

    with mlflow.start_run(run_name="random_forest"):
        # Model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
            }
        )

        # Log feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": [
                    f"feature_{i}" for i in range(len(model.feature_importances_))
                ],
                "importance": model.feature_importances_,
            }
        )
        feature_importance_path = project_root / "feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(str(feature_importance_path))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log preprocessor
        preprocessor_path = project_root / "models" / "preprocessor.joblib"
        preprocessor.save(str(preprocessor_path))
        mlflow.log_artifact(str(preprocessor_path), "preprocessor")

        # Print results
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        # Save model
        model_path = project_root / "models" / "random_forest.joblib"
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")

        return model, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }


def main():
    """Main training function."""
    print("=" * 50)
    print("Heart Disease Prediction - Model Training")
    print("=" * 50)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and fit preprocessor
    preprocessor = HeartDiseasePreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Train models
    lr_model, lr_metrics = train_logistic_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, preprocessor
    )

    rf_model, rf_metrics = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test, preprocessor
    )

    # Compare models
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    print(f"\nLogistic Regression:")
    print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {rf_metrics['roc_auc']:.4f}")

    # Select best model (based on ROC-AUC)
    if rf_metrics["roc_auc"] > lr_metrics["roc_auc"]:
        best_model = rf_model
        best_model_name = "random_forest"
        print(f"\n✓ Best model: Random Forest (ROC-AUC: {rf_metrics['roc_auc']:.4f})")
    else:
        best_model = lr_model
        best_model_name = "logistic_regression"
        print(
            f"\n✓ Best model: Logistic Regression (ROC-AUC: {lr_metrics['roc_auc']:.4f})"
        )

    # Save best model as default
    best_model_path = project_root / "models" / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
