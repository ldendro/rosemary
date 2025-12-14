"""
Fetch daily price data for a single stock (AAPL) and save it as CSV and Parquet.

Later this script will grow to handle:
- Multiple tickers
- More data sources
- Error handling and logging
"""

import os
from datetime import date

import pandas as pd
import yfinance as yf


def main() -> None:
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = str(date.today())

    raw_dir = "data/raw"
    curated_dir = "data/curated"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    print(f"Fetching daily data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    df = df.rename(columns=str.lower).reset_index()
    df["symbol"] = ticker

    csv_path = os.path.join(raw_dir, f"{ticker}_daily.csv")
    df.to_csv(csv_path, index=False)

    parquet_path = os.path.join(curated_dir, "daily_aapl.parquet")
    df.to_parquet(parquet_path, index=False)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved Parquet to {parquet_path}")
    print(f"Total rows: {len(df)}")
    
if __name__ == "__main__":
    main()
