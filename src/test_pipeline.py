"""
test_pipeline.py
End-to-end test of the MarketPredictor pipeline.
"""
import os
import subprocess

def run(cmd):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Error during execution: {cmd}")

if __name__ == "__main__":
    # Input/output files
    RAW = "data/processed/cac40_features.csv"
    CLEAN = "data/processed/cac40_features_clean.csv"
    MODEL = "models/best_model_cac40.joblib"
    SCALER = "models/best_scaler_cac40.joblib"
    PRED = "data/processed/cac40_pred.csv"

    # 1. Data preparation
    run(f"python src/data_prep.py --input {RAW} --output {CLEAN}")

    # 2. Model training
    run(f"python src/train.py --input {CLEAN} --output {MODEL} --scaler {SCALER}")

    # 3. Model evaluation
    run(f"python src/evaluate.py --data {CLEAN} --model {MODEL} --scaler {SCALER}")

    # 4. Prediction on the same data (for test)
    run(f"python src/predict.py --model {MODEL} --input {CLEAN} --output {PRED} --scaler {SCALER}")

    print("\nPipeline executed successfully. Check the generated files and logs above.")
