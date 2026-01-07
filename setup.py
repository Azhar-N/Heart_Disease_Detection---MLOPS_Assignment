from setuptools import setup, find_packages

setup(
    name="heart-disease-prediction",
    version="1.0.0",
    description="MLOps Assignment: Heart Disease Prediction System",
    author="MLOps Assignment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.9.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
    ],
)
