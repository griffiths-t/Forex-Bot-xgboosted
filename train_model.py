import pandas as pd
import joblib
from xgboost import XGBClassifier
from utils import get_candle_data, compute_indicators, is_market_open
from config import INSTRUMENT
import os

def train_model():
    print("⚙️ Training model...")
    df = get_candle_data(INSTRUMENT, granularity="M15", count=1500)

    if df.empty:
        raise ValueError("❌ No candle data returned from OANDA.")

    df = compute_indicators(df)

    feature_cols = ['rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr', 'ma_slope']
    df.dropna(inplace=True)

    if not all(col in df.columns for col in feature_cols):
        raise ValueError(f"❌ Missing required indicators: {feature_cols} not fully in dataframe")

    # Label: 1 if price increases significantly, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[feature_cols]
    y = df['target']

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("✅ Model trained and saved as model.pkl")
