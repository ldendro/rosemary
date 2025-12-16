"""
Toy backtest with transaction costs and position sizing.

This script extends the baseline backtest by introducing:
- fractional position sizing based on model confidence
- fixed transaction costs applied on position changes

This is still intentionally simple and exists to validate:
- turnover impact
- cost sensitivity
- position sizing mechanics
"""

import os
import numpy as np
import pandas as pd

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

    # -----------------------------
    # Train / test split (time-based)
    # -----------------------------
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:].copy()

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])

    test["prediction"] = model.predict(test[FEATURES])

    # -----------------------------
    # Position sizing
    # -----------------------------
    # Simple scale to normalize predictions
    scale = test["prediction"].std()
    test["position"] = (test["prediction"] / scale).clip(lower=0.0, upper=1.0)

    # -----------------------------
    # Transaction costs
    # -----------------------------
    COST_BPS = 5  # 5 basis points = 0.05%
    COST = COST_BPS / 10_000

    # Position change (turnover)
    test["prev_position"] = test.groupby("symbol")["position"].shift(1).fillna(0.0)
    test["trade_size"] = (test["position"] - test["prev_position"]).abs()

    test["transaction_cost"] = test["trade_size"] * COST

    # -----------------------------
    # Strategy return
    # -----------------------------
    test["gross_ret"] = test["position"] * test["y_ret_1d"]
    test["net_ret"] = test["gross_ret"] - test["transaction_cost"]

    # -----------------------------
    # Aggregate portfolio (equal-weight per day)
    # -----------------------------
    daily = (
        test.groupby("date", as_index=False)["net_ret"]
        .mean()
        .rename(columns={"net_ret": "portfolio_ret"})
    )

    daily["cum_log_ret"] = daily["portfolio_ret"].cumsum()
    daily["equity"] = np.exp(daily["cum_log_ret"])

    # -----------------------------
    # Metrics
    # -----------------------------
    total_return = daily["equity"].iloc[-1] - 1
    avg_daily = daily["portfolio_ret"].mean()
    vol_daily = daily["portfolio_ret"].std()
    sharpe = (avg_daily / vol_daily) * np.sqrt(252) if vol_daily > 0 else np.nan

    print("Baseline Backtest w/ Costs & Sizing")
    print("----------------------------------")
    print(f"Total return:     {total_return:.2%}")
    print(f"Avg daily return: {avg_daily:.6e}")
    print(f"Daily vol:        {vol_daily:.6e}")
    print(f"Sharpe (approx):  {sharpe:.3f}")
    print(f"Avg position:     {test['position'].mean():.3f}")
    print(f"Avg turnover:     {test['trade_size'].mean():.3f}")


if __name__ == "__main__":
    main()
