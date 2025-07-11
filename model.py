# model.py
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import compute_indicators, get_candle_data
import broker
import config

MODEL_FILE = "model.pkl"
CANDLE_COUNT = 1500
CONFIDENCE_THRESHOLD = 0.55

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

def save_model(model):
    joblib.dump(model, MODEL_FILE)

def create_features(df):
    df = compute_indicators(df)
    df["target"] = (df["close"].shift(-2) > df["close"]).astype(int)
    df = df.dropna()
    feature_cols = [
        "sma_5", "sma_15", "rsi_14", "macd", "macd_signal",
        "stoch_k", "stoch_d", "roc", "atr", "ma_slope", "hour"
    ]
    return df[feature_cols], df["target"], df

def retrain_model():
    try:
        print("[MODEL] Fetching candle data...")
        candles = broker.get_candles(instrument=config.TRADING_INSTRUMENT, count=CANDLE_COUNT)
        df = get_candle_data(candles)
        X, y, df_full = create_features(df)

        print("[MODEL] Training XGBoost classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=42, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        save_model(model)
        print("[MODEL] Model retrained and saved.")
    except Exception as e:
        print(f"[MODEL] Retraining error: {e}")
        raise

def predict_from_latest_candles():
    try:
        candles = broker.get_candles(instrument=config.TRADING_INSTRUMENT, count=50)
        df = get_candle_data(candles)
        df = compute_indicators(df)
        last_row = df.iloc[-1:]

        feature_cols = [
            "sma_5", "sma_15", "rsi_14", "macd", "macd_signal",
            "stoch_k", "stoch_d", "roc", "atr", "ma_slope", "hour"
        ]

        model = load_model()
        X_pred = last_row[feature_cols]
        proba = model.predict_proba(X_pred)[0]
        direction = int(proba[1] > proba[0])
        confidence = proba[direction]

        indicators = {col: float(last_row[col].values[0]) for col in feature_cols}
        return direction, confidence, indicators

    except Exception as e:
        print(f"[MODEL] Prediction error: {e}")
        raise