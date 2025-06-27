import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_potability_preprocessing.csv")
    data = pd.read_csv(file_path)

# Fungsi split data
def split_data(df, target='Potability', test_size=0.25, random_state=42):
    X = data.drop(columns='Potability', axis=1)
    y = data['Potability']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

X_train, X_test, y_train, y_test = split_data(data, target='Potability', test_size=0.25, random_state=42)

# Menyimpan snippet atau sample input
input_example = X_train[0:5]

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
 
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
 
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
    )
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metrics("accuracy", accuracy)