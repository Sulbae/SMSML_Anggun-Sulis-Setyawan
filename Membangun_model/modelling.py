import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import mlflow
import os
from joblib import dump

# Konfigurasi MLFLow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Water Potability Modelling with Random Forest")

# Set 2 seed untuk reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
dataset_file = "water_potability_preprocessing.csv"
dataset_path = os.path.abspath(dataset_file)
dataset_version = "v1.0"

data = pd.read_csv(dataset_file)

# Split data
X = data.drop(columns=['Potability'], axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Menyimpan snippet atau input sample
input_example = X_train.iloc[:5]

# Parameter model
n_estimators = 100
max_depth = 5

# Modelling
with mlflow.start_run():
    mlflow.sklearn.autolog(log_input_examples=True)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    # Simpan model ke file lokal
    model_path = "rf_model_v1.0.joblib"
    dump(model, model_path)

    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact(model_path, artifact_path="model_artifacts")

    # Log model setelah selesai
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )