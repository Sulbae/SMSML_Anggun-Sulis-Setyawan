import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import mlflow
import os
from joblib import dump

# Konfigurasi MLFLow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Modelling Random Forest with Grid Search")

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
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Tracking best model
best_accuracy = 0
best_params = None
best_model = None

# Modelling
with mlflow.start_run(run_name="Grid Search Experiment"):
    for params in ParameterGrid(param_grid):

        run_name = "params_" + "_".join(f"{k}{v}" for k, v in params.items())

        with mlflow.start_run(nested=True, run_name=run_name):
            # Train model
            model = RandomForestClassifier(
                **params,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_model = model
    
    # Log ke parent run
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metrics("best_accuracy", best_accuracy)
    
    # Simpan model ke file lokal
    best_model_path = "best_rf_model.joblib"
    dump(best_model, best_model_path)

    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact(best_model_path, artifact_path="best_model_artifacts")

    # Log model setelah selesai
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model"
        input_example=input_example
    )