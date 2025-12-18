import pandas as pd
import numpy as np
from pathlib import Path
import json

RAW = Path("data/raw/spy_yahoo_raw.parquet")
PROCESSED = Path("data/processed")
OUT = Path("data/outputs/diagnostics")

PROCESSED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def validate():
    df = pd.read_parquet(RAW)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    report = {}

    # Basic checks
    report["rows"] = len(df)
    report["start_date"] = str(df["date"].iloc[0].date())
    report["end_date"] = str(df["date"].iloc[-1].date())
    report["duplicate_dates"] = int(df["date"].duplicated().sum())

    # Missing values
    report["missing_adj_close"] = int(df["adj_close"].isna().sum())
    report["missing_volume"] = int(df["volume"].isna().sum())

    # Return sanity
    df["ret_1d"] = df["adj_close"].pct_change()
    extreme = df["ret_1d"].abs() > 0.20
    report["extreme_return_days"] = int(extreme.sum())

    # Calendar gaps (trading-day based)
    gaps = df["date"].diff().dt.days > 3
    report["large_date_gaps"] = int(gaps.sum())

    # Save canonical dataset
    out_df = df[["date", "adj_close", "close", "volume"]].copy()
    out_df.to_parquet(PROCESSED / "spy_daily.parquet", index=False)

    with open(OUT / "spy_validation_report_v1.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Validation complete.")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate()
