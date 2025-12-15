"""
Toy backtest for the Rosemary baseline model.

This script evaluates a very simple trading rule using the baseline model's
predictions:

Rule:
- If predicted next-day return > 0 → go long for 1 day
- Otherwise → stay flat

This backtest is intentionally naive and exists only to validate:
- alignment
- signal logic
- PnL mechanics
- absence of lookahead bias
"""

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def main() -> None:
    input_path = "data/curated/features_daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "features_daily.parquet not found. Run build_features.py first."
        )

    df = pd.read_parquet(input_path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    FEATURES = ["ret_1d", "rvol_10"]
    TARGET = "y_ret_1d"

    split_idx = int(len(df) * 0.8)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])

    test = test.copy()

    test["prediction"] = model.predict(test[FEATURES])

    test["signal"] = (test["prediction"] > 0).astype(int)

    test["strategy_ret"] = test["signal"] * test["y_ret_1d"]

    total_return = test["strategy_ret"].sum()
    avg_daily_return = test["strategy_ret"].mean()
    hit_rate = (test["strategy_ret"] > 0).mean()

    print("Toy Backtest Results")
    print("--------------------")
    print(f"Total return (log): {total_return:.6f}")
    print(f"Average daily return: {avg_daily_return:.6e}")
    print(f"Hit rate: {hit_rate:.2%}")
    print(f"Number of trades: {test['signal'].sum()}")


if __name__ == "__main__":
    main()
