import mlflow
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import argparse
import os

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Fungsi preprocess data
def preprocess_data(data, target_column, impute_method, save_path):
    # Pemisahan Fitur dan Target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Pembagian Data Latih
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        stratify=y, 
        random_state=42
    )

    # Pipeline imputasi (penanganan missing values)
    cleaning = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_method))
    ])

    # Fit pipeline untuk memproses data
    X_train = cleaning.fit_transform(X_train)
    X_test = cleaning.transform(X_test)

    # Simpan pipeline
    dump(cleaning, save_path) # save path untuk menyimpan preprocessor

    return X_train, X_test, y_train, y_test

# Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="water_potability_raw.csv")
    parser.add_argument("--target_column", type=str, default='Potability')
    parser.add_argument("--impute_method", type=str, default="median")
    parser.add_argument("--save_path", type=str, default="preprocessing/preprocessor.joblib")
    args = parser.parse_args()

    # Validasi dataset
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.dataset}")

    # Memuat data
    data = pd.read_csv(args.dataset)
    dataset_path = os.path.abspath(args.dataset)
    dataset_version = "v1.0"

    # Start MLflow run
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = preprocess_data(
            data,
            args.target_column,
            args.impute_method,
            args.save_path
        )
    
        # Log parameter
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("impute_method", args.impute_method)

        # Log preprocessor
        mlflow.log_artifact(args.save_path, artifact_path="preprocessor")
        mlflow.log_artifact(dataset_path, artifact_path="dataset")