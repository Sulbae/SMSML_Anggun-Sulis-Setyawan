import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import mlflow
import os
from joblib import dump

# Konfigurasi
DATASET_PATH = "water_potability_preprocessing.csv"
N_ESTIMATOR = 100
MAX_DEPTH = 5

# Konfigurasi MLFLow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Water Potability Modelling with Random Forest")

# Aktifkan autolog
mlflow.autolog(log_models=False, input_example=True)

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['Potability'], axis=1), 
    data['Potability'], 
    test_size=0.25, 
    random_state=42)

# Menyimpan snippet atau input sample
input_example = X_train.iloc[0:5]

# Modelling
with mlflow.start_run():
    
    # Log parameters
    mlflow.log_param("n_estimator", N_ESTIMATOR)
    mlflow.log_param("max_depth", MAX_DEPTH)

    # Train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATOR,
        max_depth=MAX_DEPTH
    )
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    # Simpan model ke file lokal
    model_path = "rf_model_v1.joblib"
    dump(model, model_path)

    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact(model_path, artifact_path="model_artifacts")

    # Log model setelah selesai
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )