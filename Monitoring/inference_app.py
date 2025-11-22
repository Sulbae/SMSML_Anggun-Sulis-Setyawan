import streamlit as st
import pandas as pd
from preprocess_prediction import data_preprocessing, prediction
import json

columns = [
        "pH", "Hardness, Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]