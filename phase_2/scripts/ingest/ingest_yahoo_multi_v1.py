from pathlib import Path
import pandas as pd
import yfinance as yf

RAW_DIR = Path("phase_2/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns (field, symbol).
    This function flattens them into single-level column names.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    return out

def ingest_one_symbol(symbol: str, start: str = "2010-01-01") -> Path:
    """
    Download one symbol from yfinance and save raw parquet:
      date, open, high, low, close, adj_close, volume
    """
    df = yf.download(symbol, start=start, auto_adjust=False, progress=False, group_by="column")

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for symbol={symbol}")

    df = _flatten_yfinance_columns(df)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index
    df = df.reset_index(drop=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    df = df[required].sort_values("date").drop_duplicates("date").reset_index(drop=True)

    out_path = RAW_DIR / f"{symbol.lower()}_yahoo_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {symbol} raw → {out_path} (rows={len(df)})")
    return out_path

def ingest_symbols(symbols: list[str], start: str = "2010-01-01") -> list[Path]:
    """
    Ingest a list of symbols and return the output paths.
    """
    outputs = []
    for sym in symbols:
        outputs.append(ingest_one_symbol(sym, start=start))
    return outputs

if __name__ == "__main__":
    symbols = ["SPY", "AAPL", "MSFT"]
    ingest_symbols(symbols, start="2010-01-01")