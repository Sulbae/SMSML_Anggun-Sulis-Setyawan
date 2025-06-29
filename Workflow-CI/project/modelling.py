import mlflow
import pandas as pd
import numpy as np
import os
import warnings
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi split data
def split_data(data, test_size=0.25, random_state=42):
    X = data.drop(columns='Potability', axis=1)
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# Script
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Argparse Parameter Input
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="project/water_potability_preprocessing.csv")
    args = parser.parse_args()

    # Validasi input
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {args.dataset}")
    
    # Memuat data
    data = pd.read_csv(args.dataset)

    # Preprocessing
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.25, random_state=42)

    # Menyimpan snippet atau sample input
    input_example = X_train.iloc[0:5]

    # Metadata dataset
    dataset_path = os.path.abspath(args.dataset)
    dataset_version = "v1.0"

    # Start MLflow run 
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log model
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )

        # Log parameters dan metrics
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", accuracy)