"""
Baseline backtest with:
- transaction costs
- position sizing
- prediction threshold
- minimum holding period
- trend filter (ret_20d > 0)
- volatility regime filter (rvol_10 <= percentile threshold)

This version adds a volatility gate to avoid high-volatility chop.
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

    FEATURES = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "rvol_10"]
    TARGET = "y_ret_1d"

    # -----------------------------
    # Train / test split
    # -----------------------------
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:].copy()

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])

    test["prediction"] = model.predict(test[FEATURES])

    # -----------------------------
    # Parameters
    # -----------------------------
    HOLD_DAYS = 5
    COST_BPS = 5
    COST = COST_BPS / 10_000

    pred_std = test["prediction"].std()
    THRESHOLD = 0.5 * pred_std

    VOL_PCTL = 0.60
    VOL_THRESHOLD = float(test["rvol_10"].quantile(VOL_PCTL))

    # -----------------------------
    # Backtest per symbol
    # -----------------------------
    records = []

    for symbol, sdf in test.groupby("symbol"):
        sdf = sdf.copy().reset_index(drop=True)

        position = 0.0
        days_left = 0

        for i in range(len(sdf)):
            row = sdf.loc[i]
            trade_cost = 0.0

            trend_ok = row["ret_20d"] > 0
            vol_ok = row["rvol_10"] <= VOL_THRESHOLD

            if days_left > 0:
                days_left -= 1

            # Entry condition (trend + volatility filters)
            if (
                position == 0.0
                and trend_ok
                and vol_ok
                and row["prediction"] > THRESHOLD
            ):
                position = min(row["prediction"] / pred_std, 1.0)
                days_left = HOLD_DAYS
                trade_cost = position * COST

            # Exit condition (unchanged)
            elif position > 0.0 and days_left == 0:
                trade_cost = position * COST
                position = 0.0

            gross_ret = position * row["y_ret_1d"]
            net_ret = gross_ret - trade_cost

            records.append(
                {
                    "date": row["date"],
                    "symbol": symbol,
                    "position": position,
                    "net_ret": net_ret,
                    "trend_ok": trend_ok,
                    "vol_ok": vol_ok,
                }
            )

    bt = pd.DataFrame(records)

    # -----------------------------
    # Aggregate portfolio
    # -----------------------------
    daily = (
        bt.groupby("date", as_index=False)["net_ret"]
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

    print("Baseline Backtest with Trend + Volatility Filter")
    print("------------------------------------------------")
    print(f"Holding days:      {HOLD_DAYS}")
    print(f"Prediction thresh: {THRESHOLD:.6e}")
    print("Trend filter:      ret_20d > 0")
    print(f"Vol filter:        rvol_10 <= {VOL_THRESHOLD:.6e} (pctl={VOL_PCTL:.2f})")
    print(f"Total return:      {total_return:.2%}")
    print(f"Avg daily return:  {avg_daily:.6e}")
    print(f"Daily vol:         {vol_daily:.6e}")
    print(f"Sharpe (approx):   {sharpe:.3f}")


if __name__ == "__main__":
    main()
