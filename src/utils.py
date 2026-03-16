def load_features(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
"""
utils.py
Utilities for saving/loading scalers and models.
"""
import joblib
import os

def save_scaler(scaler, out_path):
    joblib.dump(scaler, out_path)

def load_scaler(path):
    return joblib.load(path)
