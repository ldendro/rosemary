from pathlib import Path
import pandas as pd

RAW_DIR = Path("phase_2/data/raw")
PROCESSED_DIR = Path("phase_2/data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]

def clean_validate_daily(raw_path: Path, out_path: Path) -> None:
    """
    Clean and validate one raw Tahoo daily dataset and save canonical processed output.
    """

    df = pd.read_parquet(raw_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_path.name} missing columns: {missing}")
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    dupes = df["date"].duplicated().sum()
    if dupes > 0:
        print(f"Warning: {raw_path.name} has {dupes} duplicate dates. Dropping duplicates.")

    df = df.drop_duplicates("date")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{raw_path.name} dates are not strictly increasing.")

    day_diffs = df["date"].diff().dt.days.dropna()
    large_gaps = day_diffs[day_diffs > 7]

    if not large_gaps.empty:
        print(f"Warning: {raw_path.name} has {len(large_gaps)} large date gaps (>7 days). Largest gap: {large_gaps.max()} days")

    df = df[REQUIRED_COLUMNS]

    df.to_parquet(out_path, index=False)
    print(f"Saved processed -> {out_path} (rows={len(df)})")

def process_all_symbols(symbols: list[str]) -> None:
    for sym in symbols:
        raw_path = RAW_DIR / f"{sym.lower()}_yahoo_raw.parquet"
        out_path = PROCESSED_DIR / f"{sym.lower()}_daily.parquet"

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")
        
        clean_validate_daily(raw_path, out_path)

if __name__ == "__main__":
    symbols = ["SPY", "AAPL", "MSFT"]
    process_all_symbols(symbols)