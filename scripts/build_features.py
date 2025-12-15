"""
Build daily features for the Rosemary project.

This script reads the combined daily price table produced by the ingest script
(data/curated/daily.parquet) and computes a minimal set of foundational features:

- ret_1d: daily log return
- rvol_10: 10-day rolling volatility (annualized)
- y_ret_1d: next-day log return (supervised learning target)

Output:
- data/curated/features_daily.parquet

Later this pipeline will grow to support:
- different feature sets
- feature versioning
- more advanced signals (momentum, volatility, volume features, etc.)
"""

import os
import pandas as pd
import numpy as np


def main() -> None:
    input_path = "data/curated/daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Expected input file at {input_path}. Run the ingest script first."
        )

    df = pd.read_parquet(input_path)

    expected_cols = {"date", "open", "high", "low", "close", "adj close", "volume", "symbol"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing columns: {missing}")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)


    df["ret_1d"] = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))

    df["rvol_10"] = (
        df.groupby("symbol")["ret_1d"]
        .rolling(window=10)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)
    )

    df["y_ret_1d"] = df.groupby("symbol")["ret_1d"].shift(-1)

    df = df.dropna(subset=["ret_1d", "rvol_10", "y_ret_1d"]).reset_index(drop=True)

    output_path = "data/curated/features_daily.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Feature file written to: {output_path}")
    print(f"Total rows: {len(df)}")
    print("Feature build complete.")


if __name__ == "__main__":
    main()
