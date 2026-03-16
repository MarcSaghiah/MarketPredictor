"""
app_fastapi.py
FastAPI API to serve the MarketPredictor model.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os


app = FastAPI(title="MarketPredictor FastAPI")


# Model path dictionary for each index
MODEL_PATHS = {
    'CAC40': 'models/best_model_cac40_lightgbm_all_features.joblib',
    'SP500': 'models/best_model_sp500_lightgbm_selected_features.joblib',
}

# Cache for already loaded models
_model_cache = {}

def get_model(indice):
    if indice not in MODEL_PATHS:
        raise ValueError(f"Unsupported index: {indice}")
    if indice not in _model_cache:
        _model_cache[indice] = joblib.load(MODEL_PATHS[indice])
    return _model_cache[indice]



class InputData(BaseModel):
    data: List[dict]
    indice: str  # 'CAC40' or 'SP500'



@app.post("/predict")
def predict(input_data: InputData):
    indice = input_data.indice.upper()
    model = get_model(indice)
    df = pd.DataFrame(input_data.data)
    X = df.drop(['Target'], axis=1, errors='ignore')
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:,1]
    except AttributeError:
        y_proba = model.decision_function(X)
    result = df.copy()
    result['prediction'] = y_pred
    result['proba'] = y_proba
    return result.to_dict(orient='records')

@app.get("/")
def home():
    return {"message": "MarketPredictor FastAPI is running. Use POST /predict."}
