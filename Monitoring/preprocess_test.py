import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

preprocess_pipeline = joblib.load("preprocessing_pipeline.joblib")

def data_preprocessing(data):
    data = data.copy()    
    df = pd.DataFrame()

    preprocessed_data = preprocess_pipeline.transform(data)

    return df

if __name__ == "__main__":
    data_preprocessing()