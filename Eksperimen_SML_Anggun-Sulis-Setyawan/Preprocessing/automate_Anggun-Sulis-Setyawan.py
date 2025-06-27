import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_data(data, target_column, impute_method, save_path):
    # Pemisahan Fitur dan Target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Pembagian Data Latih
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

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