"""
train.py
Model training for MarketPredictor.
"""
import pandas as pd
import numpy as np

import joblib
import os
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from utils import save_scaler

def load_data(path):
    return pd.read_csv(path)

def train_model(X, y):
    estimators = [
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(random_state=42))
    ]
    model = VotingClassifier(estimators=estimators, voting='soft')
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.05, 0.1],
        'lgbm__num_leaves': [31, 50],
        'lgbm__learning_rate': [0.05, 0.1],
        'lgbm__max_depth': [-1, 5]
    }
    rs = RandomizedSearchCV(model, param_grid, n_iter=10, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', n_jobs=-1, random_state=42)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


def save_model(model, out_path):
    joblib.dump(model, out_path)

def save_features(feature_list, out_path):
    with open(out_path, 'w') as f:
        for feat in feature_list:
            f.write(f"{feat}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--input', type=str, required=True, help='Path to the preprocessed data file')
    parser.add_argument('--output', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--scaler', type=str, required=False, default=None, help='Path to the saved scaler')
    parser.add_argument('--features', type=str, required=False, default=None, help='Path to the saved features file')
    args = parser.parse_args()
    df = load_data(args.input)
    y = df['Target']
    X = df.drop(['Target'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model, params, score = train_model(X_scaled, y)
    save_model(model, args.output)
    if args.scaler:
        save_scaler(scaler, args.scaler)
        print(f"Scaler saved to {args.scaler}")
    if args.features:
        save_features(list(X.columns), args.features)
        print(f"Features saved to {args.features}")
    print(f"Model saved to {args.output}")
    print(f"Best parameters: {params}")
    print(f"Cross-validation score: {score:.3f}")