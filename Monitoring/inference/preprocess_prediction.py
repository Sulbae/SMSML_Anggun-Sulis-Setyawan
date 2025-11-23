import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

try:
    preprocess_pipeline = joblib.load("preprocessing_pipeline.joblib")
    print("Preprocessing pipeline berhasil dimuat.")
except FileNotFoundError as e:
    print(f"ERROR: File pipeline tidak ditemukan: {e}")
    preprocess_pipeline = None

def data_preprocessing(data_input: pd.DataFrame) -> np.ndarray:
    if preprocess_pipeline is None:
        raise RuntimeError("Preprocessing pipeline gagal dimuat.")

    preprocessed_array = preprocess_pipeline.transform(data_input)

    return preprocessed_array