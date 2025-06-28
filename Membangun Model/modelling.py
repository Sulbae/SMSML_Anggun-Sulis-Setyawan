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
mlflow.set_experiment("Water Potability Modelling with Random Forest")

# Set 2 seed untuk reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
dataset_file = "water_potability_preprocessing.csv"
dataset_path = os.path.abspath(dataset_file)
dataset_version = "v1.0"

data = pd.read_csv(dataset_file)

# Fungsi split data
def split_data(data, target='Potability', test_size=0.25, random_state=42):
    X = data.drop(columns='Potability', axis=1)
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

X_train, X_test, y_train, y_test = split_data(data, target='Potability', test_size=0.25, random_state=42)

# Menyimpan snippet atau sample input
input_example = X_train.iloc[0:5]

# Parameters model
n_estimators = 100
max_depth = 5

# Random Forest Modelling
with mlflow.start_run():
        # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    # Log parameters
    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)