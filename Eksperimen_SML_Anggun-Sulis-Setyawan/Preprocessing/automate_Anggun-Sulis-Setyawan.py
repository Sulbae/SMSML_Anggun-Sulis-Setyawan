import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(data, impute_method, save_path, output_path):
    import logging
    logging.basicConfig(level=logging.INFO)

    # Cek kondisi data
    if not data.isnull().any().any():
        logging.warning("Dataset tidak mengandung nilai NaN atau missing values.")

        # Langsung simpan data ke output_path
        data.to_csv(output_path, index=False)

        return data, output_path

    # Pipeline imputasi (penanganan missing values --jika ada)
    cleaning = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_method))
    ])

    # Fit pipeline untuk memproses data
    data_clean = cleaning.fit_transform(data)

    # Simpan pipeline
    dump(cleaning, save_path) # save path untuk menyimpan preprocessor

    # Simpan data hasil preprocessing
    cleaned_data = pd.DataFrame(data_clean, columns=data.columns)
    cleaned_data.to_csv(output_path, index=False)

    return cleaned_data, output_path