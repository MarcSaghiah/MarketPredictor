# MarketPredictor

**Worked on:** January 2026 - February 2026

**Stack:** _Python, Scikit-learn, Pandas, Plotly, yfinance, Streamlit, FastAPI_

**Directional analysis and prediction of financial markets (CAC40, S&P500, etc.) with automated pipeline, interactive dashboard, REST API.**

---

## 🚀 Overview

MarketPredictor is a complete data science project for predicting the direction of stock markets. It includes:

- A modular pipeline (preparation, training, evaluation, prediction)
- An interactive Streamlit dashboard (local or via API)
- A REST API (Flask & FastAPI)
- Code ready for production, industrialization, and portfolio showcasing

## 📂 Project Structure

```
market_predictor/
│
├── app/                # Streamlit dashboard
│   ├── app_streamlit.py
│
├── api/                # Flask & FastAPI APIs
│   ├── app_flask.py
│   └── app_fastapi.py
│
├── src/                # Pipeline scripts (data prep, train, eval, predict, utils)
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   └── test_pipeline.py
│
├── data/               # Data (example, processed)
│
├── models/             # Saved models and scalers
│
├── notebooks/          # Analysis and experimentation notebooks
│
│
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── ...
```

## ⚙️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/market_predictor.git
   cd market_predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create and activate a virtual environment.

## 🔄 Automated Pipeline

- Data preparation:

  ```bash
  python src/data_prep.py --input data/processed/cac40_features.csv --output data/processed/cac40_features_clean.csv
  ```

- Model training:

  ```bash
  python src/train.py --input data/processed/cac40_features_clean.csv --output models/best_model_cac40.joblib --scaler models/best_scaler_cac40.joblib
  ```

- Evaluation:

  ```bash
  python src/evaluate.py --data data/processed/cac40_features_clean.csv --model models/best_model_cac40.joblib --scaler models/best_scaler_cac40.joblib
  ```

- Prediction:

  ```bash
  python src/predict.py --model models/best_model_cac40.joblib --input data/processed/cac40_features_clean.csv --output data/processed/cac40_pred.csv --scaler models/best_scaler_cac40.joblib
  ```

- End-to-end test:

  ```bash
  python src/test_pipeline.py
  ```

## 📊 Interactive Streamlit Dashboard

The Streamlit dashboard provides a professional multi-index experience:

- **Select the index** (CAC40, SP500, etc.) via a dropdown menu in the sidebar
- Model and data paths are automatically adapted to the selected index (no need to enter paths manually)
- Local prediction (model) or via API (Flask/FastAPI)
- Upload a custom CSV file or use default data
- Interactive visualization: data preview, probability histograms, prediction distribution

### Launch the dashboard

```bash
streamlit run app/app_streamlit.py
```

### Usage

1. **Choose the mode**: local (model) or API (Flask/FastAPI)
2. **Select the index** from the dropdown menu (e.g., CAC40, SP500)
3. Model and data paths are automatically filled in
4. (Optional) Upload a custom CSV file for prediction
5. View results, explore distributions, download the CSV

> 💡 *By default, only CAC40 and SP500 indices are included in the project. To add a new index, simply add an entry to the `indice_options` dictionary in the code, for example:*
> ```python
> indice_options = {
>     "CAC40": {"model": "models/best_model_cac40.joblib", "data": "data/processed/cac40_features.csv"},
>     "SP500": {"model": "models/best_model_sp500.joblib", "data": "data/processed/sp500_features.csv"}
> }
> ```
> *(Replace or complete according to the indices actually available in your models/ and data/processed/ folders)*

### Main Features

- Intuitive index selection
- Local or API prediction
- Probability and class visualization
- Advanced visualization: interactive chart of historical prices (Close), period selection, overlay of predictions on the chart
- Professional and scalable user experience

## 🌐 REST API (Flask & FastAPI)

- Launch the Flask API:

  ```bash
  python api/app_flask.py
  ```

- Launch the FastAPI API:

  ```bash
  uvicorn api.app_fastapi:app --reload
  ```

## 🧑‍💻 Notebooks & Analysis

- The notebooks (notebooks/) show exploration, EDA, feature engineering, modeling, interpretation, and advanced validation (rolling window, learning curves, etc.)

## 🏆 Highlights & Best Practices

- Modular, reproducible pipeline with no data leakage
- Rolling window backtesting, learning curves, feature importance
- Interactive dashboard, REST API
- Well-commented, structured code, ready for production and portfolio showcasing
- Honest analysis of results and areas for improvement

## 📄 License

MIT

---

**Author: Marc Saghiah**
