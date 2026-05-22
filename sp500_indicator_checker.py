from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

from indicators import IndicatorConfig, build_indicator_table, summarize_latest_signal


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers(limit: int | None = None) -> list[str]:
    response = requests.get(
        SP500_WIKIPEDIA_URL,
        headers={"User-Agent": "Mozilla/5.0 sp500-indicator-checker/1.0"},
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    constituents = tables[0]

    if "Symbol" not in constituents.columns:
        raise RuntimeError("Could not find the Symbol column in the S&P 500 table.")

    tickers = (
        constituents["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return tickers[:limit] if limit else tickers


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def extract_ticker_frame(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        return downloaded[ticker].dropna(how="all")

    return downloaded.dropna(how="all")


def classify_sma(close: float, sma_short: float | None, sma_long: float | None) -> str:
    if sma_short is None or sma_long is None:
        return "insufficient_data"
    if close > sma_short > sma_long:
        return "bullish"
    if close < sma_short < sma_long:
        return "bearish"
    return "mixed"


def classify_rsi(rsi: float | None, config: IndicatorConfig) -> str:
    if rsi is None:
        return "insufficient_data"
    if rsi < config.rsi_oversold:
        return "oversold"
    if rsi > config.rsi_overbought:
        return "overbought"
    return "neutral"


def scan_sp500(
    period: str,
    interval: str,
    config: IndicatorConfig,
    batch_size: int,
    limit: int | None = None,
) -> pd.DataFrame:
    tickers = get_sp500_tickers(limit=limit)
    results: list[dict] = []

    for batch_number, tickers_batch in enumerate(chunked(tickers, batch_size), start=1):
        print(
            f"Downloading batch {batch_number}: {tickers_batch[0]} through {tickers_batch[-1]}",
            file=sys.stderr,
        )
        downloaded = yf.download(
            tickers=tickers_batch,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        for ticker in tickers_batch:
            prices = extract_ticker_frame(downloaded, ticker)
            if prices.empty:
                results.append({"ticker": ticker, "error": "no_price_data"})
                continue

            try:
                indicators = build_indicator_table(prices, config)
                summary = summarize_latest_signal(indicators)
            except Exception as exc:
                results.append({"ticker": ticker, "error": str(exc)})
                continue

            results.append(
                {
                    "ticker": ticker,
                    "close": summary["close"],
                    "rsi": summary["rsi"],
                    "rsi_state": classify_rsi(summary["rsi"], config),
                    "sma_short": summary["sma_short"],
                    "sma_long": summary["sma_long"],
                    "sma_state": classify_sma(
                        summary["close"],
                        summary["sma_short"],
                        summary["sma_long"],
                    ),
                    "golden_cross": summary["golden_cross"],
                    "death_cross": summary["death_cross"],
                    "buy_signal": summary["buy_signal"],
                    "sell_signal": summary["sell_signal"],
                    "error": "",
                }
            )

    output = pd.DataFrame(results)
    if output.empty:
        return output

    signal_order = {"bullish": 0, "oversold": 1, "overbought": 2, "mixed": 3, "bearish": 4}
    output["_sort"] = output["sma_state"].map(signal_order).fillna(99)
    output = output.sort_values(["error", "_sort", "ticker"]).drop(columns=["_sort"])
    return output.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check RSI and SMA indicators across the S&P 500."
    )
    parser.add_argument("--period", default="5y", help="Price history period for yfinance.")
    parser.add_argument("--interval", default="1d", help="Price interval for yfinance.")
    parser.add_argument("--short-sma", type=int, default=50, help="Short SMA window.")
    parser.add_argument("--long-sma", type=int, default=200, help="Long SMA window.")
    parser.add_argument("--rsi-window", type=int, default=14, help="RSI lookback window.")
    parser.add_argument("--oversold", type=float, default=30.0, help="RSI oversold threshold.")
    parser.add_argument("--overbought", type=float, default=70.0, help="RSI overbought threshold.")
    parser.add_argument("--batch-size", type=int, default=50, help="Tickers to download per request.")
    parser.add_argument("--limit", type=int, help="Scan only the first N S&P 500 tickers.")
    parser.add_argument(
        "--output",
        default="sp500_indicator_results.csv",
        help="CSV output path. Use an empty string to skip writing a CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = IndicatorConfig(
        short_sma_window=args.short_sma,
        long_sma_window=args.long_sma,
        rsi_window=args.rsi_window,
        rsi_oversold=args.oversold,
        rsi_overbought=args.overbought,
    )

    results = scan_sp500(
        period=args.period,
        interval=args.interval,
        config=config,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    if args.output:
        output_path = Path(args.output)
        results.to_csv(output_path, index=False)
        print(f"Wrote {len(results)} rows to {output_path.resolve()}")

    display_columns = [
        "ticker",
        "close",
        "rsi",
        "rsi_state",
        "sma_short",
        "sma_long",
        "sma_state",
        "golden_cross",
        "death_cross",
        "buy_signal",
        "sell_signal",
        "error",
    ]
    print(results[display_columns].to_string(index=False))
    print(f"\nSettings: {asdict(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
