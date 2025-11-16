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
N_ESTIMATORS = 100
MAX_DEPTH = 5

# Konfigurasi MLFLow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Water Potability Modelling with Random Forest")

# Aktifkan autolog
mlflow.autolog()

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Split data
X = data.drop(columns=['Potability'])
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Menyimpan snippet atau input sample
input_example = X_train.iloc[0:5]

# Modelling
with mlflow.start_run():

    # Train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH
    )
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Simpan model ke file lokal
    model_path = "rf_model_v1.joblib"
    dump(model, model_path)