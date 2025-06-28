import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(data, impute_method, save_path, output_path):
    # Pipeline imputasi (penanganan missing values)
    cleaning = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_method))
    ])

    # Fit pipeline untuk memproses data
    data_clean = cleaning.fit_transform(data)

    # Simpan pipeline
    dump(cleaning, save_path) # save path untuk menyimpan preprocessor

    # Simpan data hasil preprocessing
    cleaned_data = pd.DataFrame(data_clean, columns=data.columns)
    cleaned_data_path = output_path
    cleaned_data.to_csv(cleaned_data_path, index=False)

    return cleaned_data, cleaned_data_path