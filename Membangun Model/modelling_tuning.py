import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Membuat file ML Flow Experiment baru
mlflow.set_experiment("Random Forest Model for Water Potability")

data = pd.read_csv("water_potability_preprocessing.csv")

# Fungsi split data
def split_data(df, target='Potability', test_size=0.25, random_state=42):
    X = data.drop(columns='Potability', axis=1)
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

X_train, X_test, y_train, y_test = split_data(data, target='Potability', test_size=0.25, random_state=42)

# Menyimpan snippet atau sample input
input_example = X_train[0:5]

# Random Forest Modelling with Hyperparameter Tuning
## Elastic Search Parameters
n_estimators_range = np.linspace(10, 100, 20, dtype=int)
max_depth_range = np.linspace(5, 25, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            mlflow.autolog()

            # Latih model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluasi model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metrics("accuracy", accuracy)

            # Save model terbaik
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators:" n_estimators, "max_depth:" max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )