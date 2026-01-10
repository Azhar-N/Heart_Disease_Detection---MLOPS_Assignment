
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Add project root to path
# scripts/get_report_metrics.py -> parent = scripts -> parent = root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import load_and_preprocess_data

def main():
    print("Loading data...")
    try:
        X, y, _ = load_and_preprocess_data()
    except FileNotFoundError:
        print("Data not found. Please run 'make download' first.")
        return

    # Split matches train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load preprocessor
    preprocessor_path = project_root / "models" / "preprocessor.joblib"
    if not preprocessor_path.exists():
        print("Preprocessor not found. Run training first.")
        return
        
    preprocessor_data = joblib.load(preprocessor_path)
    # Reconstruct preprocessor manually if class method load is tricky due to imports, 
    # but we can try using the object if the class is available.
    # Actually train.py saves just the dict or object properly? 
    # train.py uses preprocessor.save(str) which dumps dict using joblib.
    # But wait, src/data/preprocessing.py defines load method.
    # Let's import the class.
    from src.data.preprocessing import HeartDiseasePreprocessor
    try:
        preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        # If it was saved as the object directly in older runs? 
        # train.py calls .save() which dumps the dict.
        # But wait, train.py:94 calls preprocessor.save()
        pass

    X_test_scaled = preprocessor.transform(X_test)

    models = ["logistic_regression", "random_forest"]
    
    results = {}

    for name in models:
        model_path = project_root / "models" / f"{name}.joblib"
        if not model_path.exists():
            print(f"Model {name} not found.")
            continue
            
        print(f"\nEvaluating {name}...")
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            "Accuracy": f"{acc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "ROC-AUC": f"{auc:.4f}"
        }
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

    print("\n--- JSON RESULTS ---")
    print(results)

if __name__ == "__main__":
    main()
