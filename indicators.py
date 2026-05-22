from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class IndicatorConfig:
    short_sma_window: int = 50
    long_sma_window: int = 200
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


def validate_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in price_data.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    cleaned = price_data.copy()
    cleaned = cleaned.sort_index()
    cleaned = cleaned.dropna(subset=["Close"])

    if cleaned.empty:
        raise ValueError("Input data contains no usable closing prices.")

    return cleaned


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is zero, RSI should read 100 instead of NaN/inf.
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def build_indicator_table(
    price_data: pd.DataFrame,
    config: Optional[IndicatorConfig] = None,
) -> pd.DataFrame:
    config = config or IndicatorConfig()
    data = validate_price_data(price_data)

    data["SMA_Short"] = calculate_sma(data["Close"], config.short_sma_window)
    data["SMA_Long"] = calculate_sma(data["Close"], config.long_sma_window)
    data["RSI"] = calculate_rsi(data["Close"], config.rsi_window)

    prev_short = data["SMA_Short"].shift(1)
    prev_long = data["SMA_Long"].shift(1)

    data["Golden_Cross"] = (
        (data["SMA_Short"] > data["SMA_Long"])
        & (prev_short <= prev_long)
    )
    data["Death_Cross"] = (
        (data["SMA_Short"] < data["SMA_Long"])
        & (prev_short >= prev_long)
    )

    data["RSI_Oversold"] = data["RSI"] < config.rsi_oversold
    data["RSI_Overbought"] = data["RSI"] > config.rsi_overbought

    data["Buy_Signal"] = data["Golden_Cross"] & data["RSI_Oversold"]
    data["Sell_Signal"] = data["Death_Cross"] & data["RSI_Overbought"]

    return data


def summarize_latest_signal(indicator_data: pd.DataFrame) -> dict:
    latest = indicator_data.iloc[-1]

    return {
        "close": float(latest["Close"]),
        "sma_short": None if pd.isna(latest["SMA_Short"]) else float(latest["SMA_Short"]),
        "sma_long": None if pd.isna(latest["SMA_Long"]) else float(latest["SMA_Long"]),
        "rsi": None if pd.isna(latest["RSI"]) else float(latest["RSI"]),
        "golden_cross": bool(latest["Golden_Cross"]),
        "death_cross": bool(latest["Death_Cross"]),
        "rsi_oversold": bool(latest["RSI_Oversold"]),
        "rsi_overbought": bool(latest["RSI_Overbought"]),
        "buy_signal": bool(latest["Buy_Signal"]),
        "sell_signal": bool(latest["Sell_Signal"]),
    }


if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"
    daily_prices = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)

    indicators = build_indicator_table(daily_prices)
    print(indicators.tail(10)[["Close", "SMA_Short", "SMA_Long", "RSI", "Buy_Signal", "Sell_Signal"]])
    print(summarize_latest_signal(indicators))
