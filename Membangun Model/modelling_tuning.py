import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import os

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
## Membuat file ML Flow Experiment baru
mlflow.set_experiment("Water Potability Modelling with Fine-Tuned Random Forest")

# Set 2 seed untuk reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
dataset_file = "water_potability_preprocessing.csv"
dataset_path = os.path.abspath(dataset_file)
dataset_version = "v1.0"

data = pd.read_csv(dataset_file)

# Fungsi split data
def split_data(data, test_size=0.25, random_state=42):
    X = data.drop(columns='Potability', axis=1)
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

X_train, X_test, y_train, y_test = split_data(data, test_size=0.25, random_state=42)

# Menyimpan snippet atau sample input
input_example = X_train.iloc[0:5]

# Random Forest Modelling with Hyperparameter Tuning
## Parameters ranges
n_estimators_range = np.linspace(10, 100, 20, dtype=int)
max_depth_range = np.linspace(5, 25, 5, dtype=int)

## Tracking best model
best_accuracy = 0
best_params = {}
best_model = None

## Grid Search
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}"):
            # Latih model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluasi model
            accuracy = model.score(X_test, y_test)

            # Log param
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("dataset_version", dataset_version)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Save model terbaik
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = model
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )