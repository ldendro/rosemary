import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

def ingest_spy(start="2010-01-01"):
    df = yf.download(
        "SPY",
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if df.empty:
        raise RuntimeError("No data downloaded for SPY")

    # -------------------------------
    # 1. Flatten MultiIndex columns
    # -------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]

    # -------------------------------
    # 2. Make date explicit
    # -------------------------------
    df = df.copy()
    df["date"] = pd.to_datetime(df.index)
    df = df.reset_index(drop=True)

    # -------------------------------
    # 3. Normalize expected columns
    # -------------------------------
    required_cols = ["date", "adj_close", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df = df[required_cols].sort_values("date")

    # -------------------------------
    # 4. Final safety checks
    # -------------------------------
    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates detected in SPY data")

    out_path = DATA_RAW / "spy_yahoo_raw.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved raw SPY data → {out_path}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    ingest_spy()
