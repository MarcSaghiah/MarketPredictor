import pandas as pd
import joblib
import os
import sys
from datetime import timedelta

 # Dynamically add src folder to sys.path for utils import
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from utils import load_features

 # Parameters
MODEL_PATH = "models/best_model_cac40_lightgbm_all_features.joblib"  # Adapt for SP500
FEATURES_PATH = "models/cac40_features_used.txt"
DATA_PATH = "data/processed/cac40_features.csv"

 # Load historical data
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date')

 # Generate the feature row for the next day
def generate_next_day_features(df):
    last_row = df.iloc[-1].copy()
    next_date = last_row['Date'] + timedelta(days=1)
    # Calculate all necessary features for the next day
    # Here, features are recalculated as in the preprocessing notebook
    # (example for moving averages, RSI, etc.)
    row = {}
    row['Date'] = next_date
    # Example features (adapt according to your pipeline)
    for window in [5, 10, 20, 50, 100, 200]:
        row[f'MA{window}'] = df['Close'].rolling(window).mean().iloc[-1]
        row[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
        row[f'Close/MA{window}'] = last_row['Close'] / row[f'MA{window}'] if row[f'MA{window}'] != 0 else 0
    row['RSI'] = df['Close'].diff().apply(lambda x: max(x,0)).rolling(14).mean().iloc[-1] / abs(df['Close'].diff()).rolling(14).mean().iloc[-1] * 100
    row['MACD'] = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1] - df['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
    row['MACD_signal'] = pd.Series([row['MACD']]).ewm(span=9, adjust=False).mean().iloc[-1]
    row['Momentum'] = last_row['Close'] - df['Close'].shift(10).iloc[-1]
    row['Volatility_5'] = df['Close'].rolling(5).std().iloc[-1]
    row['Range_5'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()).iloc[-1]
    row['Volatility_10'] = df['Close'].rolling(10).std().iloc[-1]
    row['Range_10'] = (df['High'].rolling(10).max() - df['Low'].rolling(10).min()).iloc[-1]
    row['Volatility_20'] = df['Close'].rolling(20).std().iloc[-1]
    row['Range_20'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()).iloc[-1]
    row['Return_1d'] = last_row['Close'] / df['Close'].iloc[-2] - 1 if len(df) > 1 else 0
    row['Return_5d'] = last_row['Close'] / df['Close'].iloc[-6] - 1 if len(df) > 5 else 0
    row['Return_10d'] = last_row['Close'] / df['Close'].iloc[-11] - 1 if len(df) > 10 else 0
    row['DayOfWeek'] = next_date.weekday()
    row['Day'] = next_date.day
    row['Month'] = next_date.month
    row['Year'] = next_date.year
    row['IsMonthStart'] = int(next_date.is_month_start)
    row['IsMonthEnd'] = int(next_date.is_month_end)
    # Add all other necessary features here (sector, etc.)
    # For exogenous features, you can use the last known value
    for col in df.columns:
        if col.startswith('sector_'):
            row[col] = last_row[col]
    return row

 # Generate the row for the next day
next_row = generate_next_day_features(df)

 # Load the list of expected features
with open(FEATURES_PATH) as f:
    feature_cols = [line.strip() for line in f]

 # Create the DataFrame for prediction
X_pred = pd.DataFrame([{col: next_row.get(col, 0) for col in feature_cols}])

 # Load the model and predict
model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_pred.values)
y_proba = model.predict_proba(X_pred.values)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_pred.values)

print(f"Predicted date: {next_row['Date'].strftime('%Y-%m-%d')}")
print(f"Prediction: {'Up' if y_pred[0]==1 else 'Down'}")
print(f"Probability of increase: {y_proba[0]:.2%}")
