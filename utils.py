import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_URL = "https://api-fxpractice.oanda.com/v3"

HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}"
}

SCHEDULER_LOG_PATH = "scheduler_log.txt"

def is_market_open():
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour

    if weekday == 5:
        return False
    if weekday == 6 and hour < 21:
        return False
    if weekday == 4 and hour >= 22:
        return False
    return True

def log_scheduler_message(message):
    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if now.strftime("%H:%M") == "00:00":
        with open(SCHEDULER_LOG_PATH, "w") as f:
            f.write(f"{timestamp} - {message}\n")
    else:
        with open(SCHEDULER_LOG_PATH, "a") as f:
            f.write(f"{timestamp} - {message}\n")

def format_gbp(amount):
    return f"£{amount:,.2f}"

def get_equity():
    url = f"{OANDA_URL}/accounts/{OANDA_ACCOUNT_ID}/summary"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return float(response.json()['account']['NAV'])
    return 0.0

def get_candle_data(instrument, count=1500, granularity='M15'):
    """
    Fetch candle data from OANDA API.
    Compatible with count=... usage.
    """
    url = f"{OANDA_URL}/instruments/{instrument}/candles"
    params = {
        "count": count,
        "granularity": granularity,
        "price": "M"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()

    if "candles" not in data:
        raise Exception(f"Failed to fetch candle data: {data}")

    candles = data["candles"]
    df = pd.DataFrame([{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"]),
        "volume": c["volume"]
    } for c in candles if c["complete"]])

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

def compute_indicators(df):
    df = df.copy()

    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_15"] = df["close"].rolling(window=15).mean()
    df["roc"] = df["close"].pct_change(periods=5)
    df["volatility"] = df["close"].rolling(window=14).std()
    df["ma_slope"] = df["sma_15"].diff()
    df["hour"] = df.index.hour

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Stochastic
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))