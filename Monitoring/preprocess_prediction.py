import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

def preprocess_to_predict(input=input):
    preprocess_pipeline = joblib.load("preprocessing_pipeline.joblib")

    def data_preprocessing(input):
        data = input.copy()    
        df = pd.DataFrame()

        preprocessed_data = preprocess_pipeline.transform(data)

        return df

    def prediction(data):

        url = "http://127.0.0.1:5002/invocations"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=data, headers=headers)
        response = response.json().get("predictions")
        
        return response

if __name__ == "__main__":
    preprocess_to_predict(input=input)