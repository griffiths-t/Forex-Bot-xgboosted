import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
import requests
import config
import os

SCHEDULER_LOG_PATH = "scheduler_log.txt"

def compute_indicators(df):
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_15"] = df["close"].rolling(window=15).mean()

    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    roc = ROCIndicator(close=df["close"], window=10)
    df["roc"] = roc.roc()

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"])
    df["atr"] = atr.average_true_range()

    df["ma_slope"] = df["sma_15"].diff()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour

    return df

def get_candle_data(candles):
    df = pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": int(c["volume"])
        }
        for c in candles if c["complete"]
    ])
    return df

def get_equity():
    url = f"{config.OANDA_URL}/accounts/{config.OANDA_ACCOUNT_ID}/summary"
    headers = {
        "Authorization": f"Bearer {config.OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch equity: {response.status_code} - {response.text}")
    data = response.json()
    return float(data["account"]["NAV"])

def format_gbp(amount):
    return f"£{amount:,.2f}"

def is_market_open():
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour

    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and hour < 21:  # Sunday before 21:00 UTC
        return False
    if weekday == 4 and hour >= 22:  # Friday after 22:00 UTC
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