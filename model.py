
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import broker

MODEL_PATH = "model.pkl"

def compute_features(candles_df, prefix):
    df = candles_df.copy()
    df[f'{prefix}_close'] = df['mid_c']
    df[f'{prefix}_rsi'] = df[f'{prefix}_close'].rolling(14).apply(lambda x: 100 - (100 / (1 + (np.mean(np.diff(x[x > 0])) / np.mean(np.abs(np.diff(x[x < 0]))))) if np.mean(np.abs(np.diff(x[x < 0]))) != 0 else 0), raw=False)
    df[f'{prefix}_sma_5'] = df[f'{prefix}_close'].rolling(5).mean()
    df[f'{prefix}_sma_15'] = df[f'{prefix}_close'].rolling(15).mean()
    df[f'{prefix}_roc'] = df[f'{prefix}_close'].pct_change(periods=5)
    df[f'{prefix}_volatility'] = df[f'{prefix}_close'].rolling(10).std()
    df = df.dropna().reset_index(drop=True)
    return df[[f'{prefix}_rsi', f'{prefix}_sma_5', f'{prefix}_sma_15', f'{prefix}_roc', f'{prefix}_volatility']]

def preprocess_data():
    m5 = broker.get_candles(granularity="M5", count=300)
    m15 = broker.get_candles(granularity="M15", count=300)
    h1 = broker.get_candles(granularity="H1", count=300)

    def candles_to_df(candles):
        return pd.DataFrame([{
            "time": c["time"],
            "mid_c": float(c["mid"]["c"])
        } for c in candles if c["complete"]])

    df_5m = compute_features(candles_to_df(m5), "5m")
    df_15m = compute_features(candles_to_df(m15), "15m")
    df_1h = compute_features(candles_to_df(h1), "1h")

    df = pd.concat([df_5m, df_15m, df_1h], axis=1).dropna()
    df["target"] = (df["15m_sma_5"] > df["15m_sma_15"]).astype(int)
    return df.drop(columns=["target"]), df["target"]

def retrain_model():
    X, y = preprocess_data()
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

def predict_from_latest_candles():
    X, _ = preprocess_data()
    model = joblib.load(MODEL_PATH)
    X_latest = X.iloc[[-1]]
    pred_proba = model.predict_proba(X_latest)[0][1]
    pred_class = int(pred_proba > 0.5)
    indicators = X_latest.to_dict(orient="records")[0]
    return pred_class, pred_proba, indicators
