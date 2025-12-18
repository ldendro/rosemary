"""
Fetch daily price data for a small set of stocks and save them as CSV and Parquet.

Current behavior:
- Downloads daily OHLCV data for a list of tickers (AAPL, MSFT, SPY)
- Saves one CSV per ticker under data/raw/
- Saves a single combined Parquet file under data/curated/daily.parquet

Later this script will grow to handle:
- More tickers
- Better error handling and logging
- Multiple data vendors
"""

import os
from datetime import date

import pandas as pd
import yfinance as yf


def main() -> None:
    # @anchor-ingest-tickers
    tickers = ["AAPL", "MSFT", "SPY"]
    start_date = "2015-01-01"
    end_date = str(date.today())

    raw_dir = "data/raw"
    curated_dir = "data/curated"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        print(f"Fetching daily data for {ticker} from {start_date} to {end_date}...")

        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"Warning: no data returned for {ticker}, skipping.")
            continue

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        print(f"DEBUG: Columns for {ticker} after normalization:", df.columns)
        if "date" not in df.columns:
            raise RuntimeError(f"No 'date' column found for {ticker} after processing")

        df["symbol"] = ticker

        csv_path = os.path.join(raw_dir, f"{ticker}_daily.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV for {ticker} to {csv_path}")

        all_frames.append(df)

    if not all_frames:
        print("No data was fetched for any ticker. Exiting.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    print("DEBUG: combined.columns =", combined.columns)

    if "date" in combined.columns:
        missing_dates = combined["date"].isna().sum()
        if missing_dates > 0:
            print(f"Warning: found {missing_dates} rows with missing dates.")

        duplicate_rows = combined.duplicated(subset=["symbol", "date"]).sum()
        if duplicate_rows > 0:
            print(f"Warning: found {duplicate_rows} duplicate (symbol, date) rows.")
    else:
        print("Warning: 'date' column not found in combined. Skipping date-based checks.")

    parquet_path = os.path.join(curated_dir, "daily.parquet")
    combined.to_parquet(parquet_path, index=False)

    print(f"Saved combined Parquet to {parquet_path}")
    print(f"Total rows: {len(combined)}")
    print("Ingest complete.")

if __name__ == "__main__":
    main()
