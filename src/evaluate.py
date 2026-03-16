"""
evaluate.py
Evaluation of the MarketPredictor model on a test set.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_scaler

def load_data(path):
    return pd.read_csv(path)

def load_model(path):
    return joblib.load(path)

def evaluate(model, X, y):
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:,1]
    except AttributeError:
        y_proba = model.decision_function(X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument('--data', type=str, required=True, help='Path to the test file')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--scaler', type=str, required=False, default=None, help='Path to the scaler')
    args = parser.parse_args()
    df = load_data(args.data)
    y = df['Target']
    X = df.drop(['Target'], axis=1)
    if args.scaler:
        scaler = load_scaler(args.scaler)
        X = scaler.transform(X)
    model = load_model(args.model)
    results = evaluate(model, X, y)
    print("\nEvaluation results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")