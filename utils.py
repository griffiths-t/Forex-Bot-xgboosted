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