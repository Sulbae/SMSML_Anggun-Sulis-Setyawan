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

# Random Forest Modelling
with mlflow.start_run():
    
    # Log parameters
    n_estimators = 100
    max_depth = 5
    mlflow.autolog()

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metrics("accuracy", accuracy)