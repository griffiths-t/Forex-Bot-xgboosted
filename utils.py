from datetime import datetime
import os
import requests
import pandas as pd
import config

SCHEDULER_LOG_PATH = "scheduler_log.txt"

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

def format_gbp(amount):
    return f"£{amount:,.2f}"

def get_candle_data(count=1500, granularity="M15"):
    url = f"{config.OANDA_URL}/instruments/{config.TRADING_INSTRUMENT}/candles"
    params = {
        "count": count,
        "granularity": granularity,
        "price": "M"
    }
    headers = {
        "Authorization": f"Bearer {config.OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch candles: {response.status_code} - {response.text}")

    candles = response.json()["candles"]
    data = [{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"]),
        "volume": c["volume"]
    } for c in candles if c["complete"]]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

def compute_indicators(df):
    df = df.copy()
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_15"] = df["close"].rolling(window=15).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["roc"] = df["close"].pct_change(periods=5)
    df["hour"] = df.index.hour
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))