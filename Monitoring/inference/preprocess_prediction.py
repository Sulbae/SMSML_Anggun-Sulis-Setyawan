import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json


preprocess_pipeline = joblib.load("preprocessing_pipeline.joblib")

def data_preprocessing(input):
    data = input.copy()    

    columns = [
        "pH", "Hardness, Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]

    df = pd.DataFrame([data], columns=columns)

    df = preprocess_pipeline.transform(data)

    return df