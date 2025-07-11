from datetime import datetime
import numpy as np
import pandas as pd
import os

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

    # Reset the file at 00:00 UTC
    if now.strftime("%H:%M") == "00:00":
        with open(SCHEDULER_LOG_PATH, "w") as f:
            f.write(f"{timestamp} - {message}\n")
    else:
        with open(SCHEDULER_LOG_PATH, "a") as f:
            f.write(f"{timestamp} - {message}\n")

def format_gbp(amount):
    """Format a float as British Pound currency (e.g., £1,234.56)"""
    return f"£{amount:,.2f}"

def get_candle_data(candles, timeframe="M15", count=None):
    """Converts OANDA candle API response into a clean DataFrame."""
    data = []
    for candle in candles:
        if candle['complete']:
            data.append({
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume']),
            })

    df = pd.DataFrame(data)

    if count is not None:
        df = df.tail(count)

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def compute_indicators(df):
    df = df.copy()

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    low14 = df["low"].rolling(window=14).min()
    high14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["close"] - low14) / (high14 - low14))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    # MA slope (sma_15 slope)
    df["sma_15"] = df["close"].rolling(window=15).mean()
    df["ma_slope"] = df["sma_15"].diff()

    # SMA (5)
    df["sma_5"] = df["close"].rolling(window=5).mean()

    # ROC
    df["roc"] = df["close"].pct_change(periods=5)

    # Hour of day
    df["hour"] = df.index.hour

    return df.dropna()