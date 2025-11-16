import mlflow
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import numpy as np

# Konfigurasi 
DATASET_PATH = "water_potability_preprocessing.csv"
N_ESTIMATORS_RANGE = np.linspace(10, 100, 3, dtype=int)
MAX_DEPTH_RANGE = np.linspace(1, 50, 3, dtype=int)
TEST_SIZE = 0.25

# Set Tracking melalui DagsHub
dagshub.init(
    repo_owner="Sulbae",
    repo_name="Latihan-MLFlow",
    mlflow=True
)
# Set Nama Eksperimen
mlflow.set_experiment("Modelling Random Forest with Grid Search")

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Split data
X = data.drop(columns=['Potability'], axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, 
    random_state=42)

# Menyimpan snippet atau input sample
input_example = X_train.iloc[0:5]

best_acc = 0
best_params = {}
best_model = None

for n_estimators in N_ESTIMATORS_RANGE:
    for max_depth in MAX_DEPTH_RANGE:
        with mlflow.start_run(run_name=f"grid_search_{n_estimators}_{max_depth}") as run:

            # Log Parameter
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("test_size", TEST_SIZE)

            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log Metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Save the best model
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = {
                    "n_estimators": n_estimators, 
                    "max_depth": max_depth
                }
                best_model = model

with mlflow.start_run(run_name="best_model_run"):
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_acc)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        input_example=input_example
    )