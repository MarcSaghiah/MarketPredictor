"""
predict.py
Script d'inférence pour MarketPredictor.
"""

import pandas as pd
import joblib
from utils import load_scaler

def load_model(path):
    return joblib.load(path)

def load_data(path):
    return pd.read_csv(path)

def predict(model, X):
    try:
        y_proba = model.predict_proba(X)[:,1]
    except AttributeError:
        y_proba = model.decision_function(X)
    y_pred = model.predict(X)
    return y_pred, y_proba

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prédiction avec MarketPredictor")
    parser.add_argument('--model', type=str, required=True, help='Chemin du modèle')
    parser.add_argument('--input', type=str, required=True, help='Chemin du fichier de données')
    parser.add_argument('--output', type=str, required=True, help='Chemin du fichier de sortie')
    parser.add_argument('--scaler', type=str, required=False, default=None, help='Chemin du scaler')
    args = parser.parse_args()
    model = load_model(args.model)
    df = load_data(args.input)
    X = df.drop(['Target'], axis=1, errors='ignore')
    if args.scaler:
        scaler = load_scaler(args.scaler)
        X = scaler.transform(X)
    y_pred, y_proba = predict(model, X)
    df['prediction'] = y_pred
    df['proba'] = y_proba
    df.to_csv(args.output, index=False)
    print(f"Prédictions sauvegardées sous {args.output}")