# =====================================================
# Imports and configuration
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import requests
import yfinance as yf
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import plotly.express as px

# Dynamically add src folder to sys.path for utils import
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Import project utilities from src/utils.py
import importlib.util
utils_path = os.path.join(src_path, 'utils.py')
spec = importlib.util.spec_from_file_location('utils', utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
load_features = utils.load_features

# =====================================================
# Utility functions
# =====================================================

@st.cache_data(show_spinner=False)
def fetch_market_data(ticker, period="max"):
    """
    Fetch market data from Yahoo Finance for a given ticker and period.
    """
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model_and_features(model_path):
    """
    Load a model and its feature list from disk.
    """
    model = joblib.load(model_path)
    features_used_path = None
    if "cac40" in model_path.lower():
        features_used_path = "models/cac40_features_used.txt"
    elif "sp500" in model_path.lower():
        features_used_path = "models/sp500_features_used.txt"
    feature_cols = None
    if features_used_path and os.path.exists(features_used_path):
        with open(features_used_path) as f:
            feature_cols = [line.strip() for line in f]
        feature_cols = [col for col in feature_cols if col not in ('Target', 'Date')]
    return model, feature_cols

def prepare_features_local(df, feature_cols):
    """
    Prepare local features for prediction from dataframe.
    """
    if feature_cols is not None:
        X_df = pd.DataFrame({col: df[col] if col in df.columns else [0]*len(df) for col in feature_cols})[feature_cols]
        X = X_df.values
    else:
        exclude_cols = ['Target', 'Date']
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        X_df = df[feature_cols]
    return X_df

def prepare_features_api(df, indice):
    """
    Prepare features for API prediction from dataframe.
    """
    df_api = df.copy()
    df_api.columns = [str(col) if not isinstance(col, str) else col for col in df_api.columns]
    exclude_cols = ['Target', 'Date']
    features_used_path = None
    if indice.upper() == "CAC40":
        features_used_path = "models/cac40_features_used.txt"
    elif indice.upper() == "SP500":
        features_used_path = "models/sp500_features_used.txt"
    if features_used_path and os.path.exists(features_used_path):
        with open(features_used_path) as f:
            feature_cols = [line.strip() for line in f]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        last_row = df_api.iloc[[-1]].copy()
        last_row = pd.DataFrame({col: last_row[col] if col in last_row.columns else [0] for col in feature_cols})[feature_cols]
        df_api = last_row
    else:
        feature_cols = [col for col in df_api.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_api[col])]
        df_api = df_api.iloc[[-1]][feature_cols]
    return df_api

def predict_local(model, X_df):
    """
    Predict locally using the loaded model.
    """
    y_pred = model.predict(X_df)
    y_proba = model.predict_proba(X_df)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_df)
    return y_pred, y_proba

def predict_api(df_api, indice, api_url):
    """
    Predict using external API.
    """
    data_json = df_api.to_dict(orient='records')
    payload = {"data": data_json, "indice": indice}
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            try:
                api_result = response.json()
                if isinstance(api_result, dict):
                    pred_df = pd.DataFrame([api_result])
                else:
                    pred_df = pd.DataFrame(api_result)
                if 'prediction' in pred_df.columns and 'proba' in pred_df.columns:
                    return pred_df['prediction'].iloc[0], pred_df['proba'].iloc[0]
                else:
                    st.error(f"Réponse API inattendue : colonnes manquantes. Contenu : {pred_df}")
            except Exception as e:
                st.error(f"Erreur de parsing JSON ou format inattendu : {e}\nContenu brut : {response.text}")
                st.stop()
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        st.stop()
    return None, None

def plot_index_chart(df):
    """
    Plot the main index chart.
    """
    close_col = None
    for col in df.columns:
        if (isinstance(col, tuple) and str(col[0]).lower() == 'close') or (isinstance(col, str) and col.lower() == 'close'):
            close_col = col
            break
    if close_col is None or 'Date' not in df.columns:
        st.warning("Unable to display chart: 'Close' or 'Date' column missing from data. (Fallback chart displayed)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,0], mode='lines', name='Dummy'))
        fig.update_layout(title="No data to display", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    close_series = df[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = pd.to_numeric(close_series, errors='coerce')
    df_graph = pd.DataFrame({'Date': df['Date'], 'Close': close_series}).dropna(subset=['Date', 'Close']).sort_values('Date')
    if not df_graph.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_graph['Date'],
            y=df_graph['Close'],
            mode='lines',
            name='Closing price',
            line=dict(color='royalblue')
        ))
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No valid data to display for chart. (Fallback chart displayed)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,0], mode='lines', name='Dummy'))
        fig.update_layout(title="No data to display", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')
    return df_graph

def display_prediction_section(df_graph):
    """
    Display prediction section for selected date.
    """
    if 'prediction' in df_graph.columns and 'proba' in df_graph.columns:
        st.markdown("<div style='margin-bottom:0rem;'></div>", unsafe_allow_html=True)
        st.subheader("Up/Down prediction for a date")
        last_row = df_graph.iloc[-1]
        date_selected = st.selectbox("Select date to predict", options=df_graph['Date'].dt.strftime('%Y-%m-%d'), index=len(df_graph)-1)
        row_pred = df_graph[df_graph['Date'].dt.strftime('%Y-%m-%d') == date_selected].iloc[0]
        pred_value = row_pred['prediction']
        if isinstance(pred_value, pd.Series):
            pred_value = pred_value.values[0]
            sens = 'Up' if pred_value == 1 else 'Down'
        proba = row_pred['proba']
        if isinstance(proba, pd.Series):
            proba = proba.values[0]
        date_value = row_pred['Date']
        if isinstance(date_value, pd.Series):
            date_value = date_value.values[0]
        if isinstance(date_value, (pd.Timestamp, datetime.datetime)):
            date_str = date_value.strftime('%Y-%m-%d')
        elif isinstance(date_value, (np.datetime64)):
            date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        else:
            date_str = str(date_value)
            st.markdown(f"**Date:** {date_str}  ")
            st.markdown(f"**Prediction:** {sens}")
            st.markdown(f"**Probability of increase:** {proba:.2%}")
    else:
        # Nothing
        pass

def build_next_day_features(df_graph, last_row):
    """
    Build features for next-day prediction.
    """
    # Find the next business day (Monday to Friday) from the most recent date between the last data and today
    import datetime
    today = datetime.date.today()
    last_data_date = last_row['Date'].date() if hasattr(last_row['Date'], 'date') else pd.to_datetime(last_row['Date']).date()
    base_date = max(today, last_data_date)
    next_date = base_date + timedelta(days=1)
    # Skip weekends
    while next_date.weekday() >= 5:  # 5 = samedi, 6 = dimanche
        next_date += timedelta(days=1)
    if isinstance(next_date, pd.Series):
        next_date = next_date.values[0]
    if isinstance(next_date, np.datetime64):
        next_date = pd.to_datetime(next_date)
    row = {'Date': next_date}
    for window in [5, 10, 20, 50, 100, 200]:
        ma_val = df_graph['Close'].rolling(window).mean().iloc[-1]
        if isinstance(ma_val, (pd.Series, np.ndarray)):
            ma_val = ma_val.item() if hasattr(ma_val, 'item') else float(ma_val)
        row[f'MA{window}'] = ma_val
        row[f'EMA{window}'] = df_graph['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
        row[f'Close/MA{window}'] = float(last_row['Close']) / float(ma_val) if ma_val != 0 else 0
    try:
        rsi_num = df_graph['Close'].diff().apply(lambda x: max(x,0)).rolling(14).mean().iloc[-1]
        rsi_den = abs(df_graph['Close'].diff()).rolling(14).mean().iloc[-1]
        row['RSI'] = (rsi_num / rsi_den) * 100 if rsi_den != 0 else 0
    except Exception:
        row['RSI'] = 0
    row['MACD'] = df_graph['Close'].ewm(span=12, adjust=False).mean().iloc[-1] - df_graph['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
    row['MACD_signal'] = pd.Series([row['MACD']]).ewm(span=9, adjust=False).mean().iloc[-1]
    row['Momentum'] = last_row['Close'] - df_graph['Close'].shift(10).iloc[-1] if len(df_graph) > 10 else 0
    high_col = None
    low_col = None
    for col in df_graph.columns:
        if (isinstance(col, tuple) and str(col[0]).lower() == 'high') or (isinstance(col, str) and col.lower() == 'high'):
            high_col = col
            st.warning("Unable to display chart: 'Close' or 'Date' column missing from data. (Fallback chart displayed)")
            low_col = col
    for w in [5, 10, 20]:
        if high_col is not None and low_col is not None:
            row[f'Range_{w}'] = (df_graph[high_col].rolling(w).max() - df_graph[low_col].rolling(w).min()).iloc[-1]
        else:
            row[f'Range_{w}'] = 0
    row['Return_1d'] = last_row['Close'] / df_graph['Close'].iloc[-2] - 1 if len(df_graph) > 1 else 0
    row['Return_5d'] = last_row['Close'] / df_graph['Close'].iloc[-6] - 1 if len(df_graph) > 5 else 0
    row['Return_10d'] = last_row['Close'] / df_graph['Close'].iloc[-11] - 1 if len(df_graph) > 10 else 0
    row['DayOfWeek'] = next_date.weekday()
    row['Day'] = next_date.day
    row['Month'] = next_date.month
    row['Year'] = next_date.year
    row['IsMonthStart'] = int(getattr(next_date, 'is_month_start', False))
    row['IsMonthEnd'] = int(getattr(next_date, 'is_month_end', False))
    for col in df_graph.columns:
        if isinstance(col, str) and col.startswith('sector_'):
            row[col] = last_row[col]
    return row

def display_next_day_prediction(df_graph, mode, model, model_path, indice, api_url):
    """
    Display next-day prediction results.
    """
    col1, col2 = st.columns([1,3])
    result_message = ""
    if 'next_day_pred' not in st.session_state:
        st.session_state['next_day_pred'] = None
    with col1:
        predict_clicked = st.button("Next-day prediction")
    with col2:
        if predict_clicked:
            last_row = df_graph.iloc[-1].copy()
            row = build_next_day_features(df_graph, last_row)
            features_used_path = None
            if mode == "Local (model)":
                if model_path and "cac40" in model_path.lower():
                    features_used_path = "models/cac40_features_used.txt"
                elif model_path and "sp500" in model_path.lower():
                    features_used_path = "models/sp500_features_used.txt"
            else:
                if indice and indice.upper() == "CAC40":
                    features_used_path = "models/cac40_features_used.txt"
                elif indice and indice.upper() == "SP500":
                    features_used_path = "models/sp500_features_used.txt"
            if features_used_path and os.path.exists(features_used_path):
                with open(features_used_path) as f:
                    feature_cols = [line.strip() for line in f]
                feature_cols = [col for col in feature_cols if col not in ('Target', 'Date')]
                X_pred = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])
                if mode == "Local (model)":
                    if model is not None:
                        y_pred_next = model.predict(X_pred.values)
                        y_proba_next = model.predict_proba(X_pred.values)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_pred.values)
                        sens_next = 'Up' if y_pred_next[0] == 1 else 'Down'
                        result_message = f"Date: {row['Date'].strftime('%d-%m-%Y')} — Prediction: {sens_next} (probability of increase: {y_proba_next[0]:.2%})"
                        st.session_state['next_day_pred'] = result_message
                    else:
                        result_message = "Model not loaded for local next-day prediction."
                        st.session_state['next_day_pred'] = result_message
                else:
                    data_json = X_pred.to_dict(orient='records')
                    payload = {"data": data_json, "indice": indice}
                    try:
                        response = requests.post(api_url, json=payload)
                        if response.status_code == 200:
                            try:
                                api_result = response.json()
                                if isinstance(api_result, dict):
                                    pred_df = pd.DataFrame([api_result])
                                else:
                                    pred_df = pd.DataFrame(api_result)
                                if 'prediction' in pred_df.columns and 'proba' in pred_df.columns:
                                    sens_next = 'Up' if pred_df['prediction'].iloc[0] == 1 else 'Down'
                                    proba_next = pred_df['proba'].iloc[0]
                                    result_message = f"Predicted date: {row['Date'].strftime('%Y-%m-%d')} — Prediction: {sens_next} (probability of increase: {proba_next:.2%})"
                                    st.session_state['next_day_pred'] = result_message
                                else:
                                    result_message = f"Unexpected API response for next-day prediction: missing columns. Content: {pred_df}"
                                    st.session_state['next_day_pred'] = result_message
                            except Exception as e:
                                result_message = f"JSON parsing error or unexpected format for next-day prediction: {e}\nRaw content: {response.text}"
                                st.session_state['next_day_pred'] = result_message
                        else:
                            result_message = f"API error (next day): {response.status_code} - {response.text}"
                            st.session_state['next_day_pred'] = result_message
                    except Exception as e:
                        result_message = f"Error during API call for next-day prediction: {e}"
                        st.session_state['next_day_pred'] = result_message
            else:
                result_message = "Unable to load feature list for next-day prediction."
                st.session_state['next_day_pred'] = result_message
        # Display result block (even if not clicked)
        if st.session_state['next_day_pred']:
            st.success(st.session_state['next_day_pred'])
        else:
            st.markdown('<div style="min-height:48px;"></div>', unsafe_allow_html=True)

# =====================
# MAIN DASHBOARD
# =====================
def main():

    # Modern header and global style to reduce padding
    st.markdown("""
        <style>
            .block-container { padding-top: 0.15rem !important; padding-bottom: 0.15rem !important; }
            .element-container, .stPlotlyChart, .stMarkdown, .stButton, .stSelectbox, .stSubheader {
                margin-top: 0rem !important;
                margin-bottom: 0rem !important;
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }
            hr { margin-top: 0.1rem !important; margin-bottom: 0.2rem !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='padding-top: 2rem;'></div>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <span style='font-size:2.5rem;'>📈</span>
            <div>
                <span style='font-size:2.1rem; font-weight:700; color:#1a4e8a;'>MarketPredictor</span><br>
            </div>
        </div>
        <hr style='margin-top:0.2rem; margin-bottom:0.7rem;'>
    """, unsafe_allow_html=True)


    # Sidebar: Help & Disclaimer only
    with st.sidebar:
        st.markdown("""
            <div style='text-align:center; margin-bottom:0.5rem;'>
                <span style='font-size:2.2rem;'>📈 MarketPredictor</span>
            </div>
            <hr style='margin-top:0.1rem; margin-bottom:0.7rem;'>
        """, unsafe_allow_html=True)
        # Prediction mode in the sidebar
        mode = st.radio("Prediction mode", ["Local (model)", "API (Flask/FastAPI)"], format_func=lambda x: "🔗 API" if "API" in x else "💻 Local", key="mode_radio_sidebar")
        st.markdown("""
            <span style='font-size:1.1rem; font-weight:500;'>Data source</span>
        """, unsafe_allow_html=True)
        use_live_data = st.checkbox("Use live Yahoo Finance data", value=True, key="live_data_main")
        st.markdown("""
            <hr style='margin-top:1.2rem; margin-bottom:0.7rem;'>
            <span style='font-size:1.1rem; font-weight:500;'>Help & Disclaimer</span>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ Documentation & FAQ", expanded=False):
            st.markdown("""
                **MarketPredictor is a dashboard for predicting stock index trends using historical data and technical indicators.**

                **How to use this dashboard?**  
                1. Select the index, period, and chart to display in the main window.  
                2. Click the prediction button to get the expected trend for the next trading day.  
                3. Results are for informational purposes only.

                **Methodology**  
                Prediction based on a LightGBM model, trained on historical Yahoo Finance data. Features used include classic technical indicators (moving averages, RSI, MACD, etc.).
            """)
        with st.expander("📊 Graph Descriptions", expanded=False):
            st.markdown("""
                <ul style='font-size:0.97rem; margin-top:0.3rem;'>
                    <li><b>Index price</b>: evolution of the closing price of the selected index over the chosen period.</li>
                    <li><b>Traded volume</b>: total quantity of securities traded each day (market liquidity).</li>
                    <li><b>RSI</b>: relative strength indicator, measures market momentum (overbought/oversold).</li>
                    <li><b>MACD</b>: trend indicator based on the difference of exponential moving averages.</li>
                    <li><b>Returns histogram</b>: distribution of daily closing price changes.</li>
                    <li><b>Statistics table</b>: descriptive statistics on the closing price (mean, min, max, etc.).</li>
                </ul>
            """, unsafe_allow_html=True)
        with st.expander("⚠️ Disclaimer", expanded=False):
            st.markdown("""
                This dashboard is an experimental data science project.  
                **No guarantee of performance, accuracy, or reliability is given.**  
                This is not a professional, banking, or financial tool.  
                Do not make any investment decisions based on these results.  
                Investing in the stock market involves risk of capital loss.  
                The author declines all responsibility for the use of predictions or displayed data.
            """)

    # Main window: configuration and display
    st.markdown("""
        <div style='display: flex; gap: 2rem; align-items: flex-end;'>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,2,3])
    with col1:
        indice_options = {
            "CAC40": {
                "model": "models/best_model_cac40_lightgbm_all_features.joblib",
                "ticker": "^FCHI"
            },
            "SP500": {
                "model": "models/best_model_sp500_lightgbm_selected_features.joblib",
                "ticker": "^GSPC"
            }
        }
        indice = st.selectbox("Index", list(indice_options.keys()), format_func=lambda x: "🇫🇷 CAC40" if x=="CAC40" else "🇺🇸 S&P500", key="indice_select_main")
    with col2:
        # Choice of period type
        today = datetime.date.today()
        min_date = datetime.date(2000, 1, 1)
        one_week_ago = today - datetime.timedelta(weeks=1)
        one_month_ago = today - datetime.timedelta(days=30)
        three_months_ago = today - datetime.timedelta(days=90)
        six_months_ago = today - datetime.timedelta(days=182)
        one_year_ago = today.replace(year=today.year-1)
        two_years_ago = today.replace(year=today.year-2)
        five_years_ago = today.replace(year=today.year-5)
        ten_years_ago = today.replace(year=today.year-10)
        period_options = {
            "Full period": [min_date, today],
            "Last week": [one_week_ago, today],
            "Last month": [one_month_ago, today],
            "Last 3 months": [three_months_ago, today],
            "Last 6 months": [six_months_ago, today],
            "Last year": [one_year_ago, today],
            "Last 2 years": [two_years_ago, today],
            "Last 5 years": [five_years_ago, today],
            "Last 10 years": [ten_years_ago, today],
            "Custom": None
        }
        default_period = "Full period"
        # Previous selection detection
        previous_range = st.session_state.get("date_range_main", [min_date, today])
        if previous_range == [five_years_ago, today]:
            default_period = "Last 5 years"
        elif previous_range == [ten_years_ago, today]:
            default_period = "Last 10 years"
        elif previous_range != [min_date, today]:
            default_period = "Custom"
        period_choice = st.selectbox(
            "Analysis period",
            list(period_options.keys()),
            index=list(period_options.keys()).index(default_period),
            key="period_choice_main"
        )
    with col3:
        # If custom, display the date selector in range mode (start/end)
        if period_choice == "Custom":
            # Explicit selection: separate start and end dates
            if not (isinstance(previous_range, (list, tuple)) and len(previous_range) == 2):
                previous_range = [min_date, today]
            # Correction if one of the two dates is None or empty
            if previous_range[0] is None:
                previous_range[0] = min_date
            if previous_range[1] is None:
                previous_range[1] = today
            col_debut, col_fin = st.columns(2)
            with col_debut:
                start_date = st.date_input(
                    label="Start date",
                    value=previous_range[0],
                    min_value=min_date,
                    max_value=today,
                    key="date_debut_main"
                )
            with col_fin:
                end_date = st.date_input(
                    label="End date",
                    value=previous_range[1],
                    min_value=min_date,
                    max_value=today,
                    key="date_fin_main"
                )
            if start_date > end_date:
                st.warning("Start date must be earlier than or equal to end date.")
                st.stop()
            date_range = [start_date, end_date]
            # Always synchronize the session with the displayed selection
            st.session_state["date_range_main"] = date_range
        else:
            date_range = period_options[period_choice]
            st.session_state["date_range_main"] = date_range
    st.markdown("</div>", unsafe_allow_html=True)

    # Main chart tabs (pro UX)
    tab_labels = [
        "Index price",
        "Traded volume",
        "RSI",
        "MACD",
        "Returns histogram",
        "Statistics table"
    ]
    tabs = st.tabs(tab_labels)

    # mode and use_live_data are now in the sidebar
    # End main configuration

    # Get the selected mode from the sidebar
    mode = st.session_state.get("mode_radio_sidebar", "Local (model)")
    use_live_data = st.session_state.get("live_data_main", True)
    if mode == "Local (model)":
        model_path = indice_options[indice]["model"]
        api_url = None
    else:
        api_url = "http://localhost:8000/predict"
        model_path = None

    if use_live_data:
        df = fetch_market_data(indice_options[indice]["ticker"], period="max")
        if df.empty:
            st.warning("Unable to retrieve Yahoo Finance data.")
            st.stop()
    else:
        uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("No data available.")
            st.stop()

    # Prediction
    if mode == "Local (model)":
        try:
            model, feature_cols = load_model_and_features(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        X = prepare_features_local(df, feature_cols)
        try:
            y_pred, y_proba = predict_local(model, X)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()
        df['prediction'] = y_pred
        df['proba'] = y_proba
    else:
        df_api = prepare_features_api(df, indice)
        y_pred, y_proba = predict_api(df_api, indice, api_url)
        df.at[df.index[-1], 'prediction'] = y_pred
        df.at[df.index[-1], 'proba'] = y_proba

    # Robust cleaning and filtering after loading
    # Smart extraction of Date and Close columns (even if multi-index or tuple)
    # Find the Date column
    date_col = None
    for col in df.columns:
        if (isinstance(col, str) and col.lower() == 'date') or (isinstance(col, tuple) and str(col[0]).lower() == 'date'):
            date_col = col
            break
    close_col = None
    for col in df.columns:
        if (isinstance(col, str) and col.lower() == 'close') or (isinstance(col, tuple) and str(col[0]).lower() == 'close'):
            close_col = col
            break
    if date_col is None or close_col is None:
        st.error(f"Unable to find 'Date' and 'Close' columns in columns: {list(df.columns)}")
        st.stop()
        st.warning("No valid data to display for chart. (Fallback chart displayed)")
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    close_series = df[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    elif isinstance(close_series, (list, tuple, np.ndarray)):
        close_series = pd.Series(close_series)
    if hasattr(close_series, 'apply'):
        close_series = close_series.apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    if not isinstance(close_series, pd.Series):
        st.error(f"Unable to interpret Close column for numeric conversion: type {type(close_series)}")
        st.stop()
    df['Close'] = pd.to_numeric(close_series, errors='coerce')
    # Strict check for required columns
    required_cols = ['Date', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns after loading/filtering: {missing_cols}.\nAvailable columns: {list(df.columns)}")
        st.stop()
    # Filter by period
    if not date_range or not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.error("No valid date range selected. Please select a period.")
        st.stop()
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    # Keep only columns useful for display (simple columns)
    # Flatten multi-index columns to simple strings
    def flatten_col(col):
        if isinstance(col, tuple):
            return str(col[0]) if len(col) > 0 else str(col)
        return str(col)
    df_flat = df.copy()
    df_flat.columns = [flatten_col(col) for col in df_flat.columns]
    # Extraction and renaming on flattened columns
    col_map = {}
    for col in df_flat.columns:
        if col.lower() == 'date':
            col_map[col] = 'Date'
        if col.lower() == 'close':
            col_map[col] = 'Close'
        if col.lower() == 'prediction':
            col_map[col] = 'prediction'
        if col.lower() == 'proba':
            col_map[col] = 'proba'
    if not ('Date' in col_map.values() and 'Close' in col_map.values()):
        st.error(f"Unable to find 'Date' and 'Close' columns for display. Available columns: {list(df_flat.columns)}")
        st.stop()
    df_graph = df_flat[list(col_map.keys())].copy()
    df_graph = df_graph.rename(columns=col_map)
    # Dropna only on this reduced DataFrame
    required_cols = ['Date', 'Close']
    missing_cols = [col for col in required_cols if col not in df_graph.columns]
    if missing_cols:
        st.error(f"Missing columns just before dropna: {missing_cols}.\nAvailable columns: {list(df_graph.columns)}")
        st.stop()
    try:
        df_graph = df_graph.dropna(subset=required_cols)
    except KeyError as e:
        st.error(f"KeyError during dropna: {e}.\nDataFrame columns: {list(df_graph.columns)}")
        st.stop()

    # Strict check and fallback chart if needed
    if df_graph.empty or df_graph['Date'].isnull().all() or df_graph['Close'].isnull().all():
        st.warning("No usable data for charts. (Fallback chart displayed)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,0], mode='lines', name='Dummy'))
        fig.update_layout(title="No data to display", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')
        # Continue execution to keep the UI alive
    if df_graph['Close'].nunique() <= 1:
        st.warning("Warning: the 'Close' column contains only one unique value or all values are identical. Charts may be flat or invisible.")
    # Date_fmt column for display only
    df['Date_fmt'] = df['Date'].dt.strftime('%d-%m-%Y')

    # Main display with tabs
    if all(col in df_graph.columns for col in ['Date', 'Close']):
        model_val = locals().get('model', None)
        model_path_val = locals().get('model_path', None)
        api_url_val = locals().get('api_url', None)

        with tabs[0]:
            # Index price
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_graph['Date'],
                y=df_graph['Close'],
                mode='lines',
                name='Closing price',
                line=dict(color='royalblue'),
                hovertemplate='%{y}<br>%{x|%d-%m-%Y}'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Closing price",
                margin=dict(t=20, b=10, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig, width='stretch')

        with tabs[1]:
            # Traded volume
            if 'Volume' in df_graph.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=df_graph['Date'], y=df_graph['Volume'], marker_color='rgba(100,100,200,0.5)', name='Volume',
                                         hovertemplate='%{y}<br>%{x|%d-%m-%Y}'))
                fig_vol.update_layout(title='Traded volume', xaxis_title='Date', yaxis_title='Volume', margin=dict(t=20, b=10, l=10, r=10), height=350)
                st.plotly_chart(fig_vol, width='stretch')
            else:
                st.info("No volume data available for this index.")

        with tabs[2]:
            # RSI
            window = 14
            delta = df_graph['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=window).mean()
            loss = -delta.clip(upper=0).rolling(window=window).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi = 100 - (100 / (1 + rs))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_graph['Date'], y=rsi, mode='lines', name='RSI', line=dict(color='orange'),
                                         hovertemplate='%{y:.2f}<br>%{x|%d-%m-%Y}'))
            fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
            fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
            fig_rsi.update_layout(title='RSI (14 days)', xaxis_title='Date', yaxis_title='RSI', margin=dict(t=20, b=10, l=10, r=10), height=350)
            st.plotly_chart(fig_rsi, width='stretch')

        with tabs[3]:
            # MACD
            ema12 = df_graph['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_graph['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_graph['Date'], y=macd, mode='lines', name='MACD', line=dict(color='blue'),
                                          hovertemplate='%{y:.2f}<br>%{x|%d-%m-%Y}'))
            fig_macd.add_trace(go.Scatter(x=df_graph['Date'], y=signal, mode='lines', name='Signal', line=dict(color='red'),
                                          hovertemplate='%{y:.2f}<br>%{x|%d-%m-%Y}'))
            fig_macd.update_layout(title='MACD', xaxis_title='Date', yaxis_title='MACD', margin=dict(t=20, b=10, l=10, r=10), height=350)
            st.plotly_chart(fig_macd, width='stretch')

        with tabs[4]:
            # Returns histogram
            returns = df_graph['Close'].pct_change().dropna()
            fig_hist = px.histogram(returns, nbins=50, title='Distribution of daily returns')
            fig_hist.update_layout(xaxis_title='Daily return', yaxis_title='Frequency', margin=dict(t=20, b=10, l=10, r=10), height=350)
            st.plotly_chart(fig_hist, width='stretch')

        with tabs[5]:
            # Statistics table
            stats = df_graph['Close'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
            # Display with explicit column name
            stats_df = pd.DataFrame(stats)
            stats_df.columns = ['Closing price']
            st.dataframe(stats_df)

        # Compact prediction section (always visible below the tabs)
        display_next_day_prediction(df_graph, mode, model_val, model_path_val, indice, api_url_val)

    else:
        st.warning("Required columns (Date, Close, prediction, proba) are not all present in the data.")

if __name__ == "__main__":
    main()