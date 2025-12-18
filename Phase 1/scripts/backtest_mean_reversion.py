"""
Rule-based Mean Reversion Backtest (single-split baseline).

Strategy (per symbol):
- Enter long if ret_5d < -THRESH
- Hold for HOLD_DAYS
- Apply transaction costs on entry and exit
- Position size is fixed at 1.0 (no leverage)

This is intentionally simple and designed to complement a trend strategy.
"""

import os
import numpy as np
import pandas as pd


def main() -> None:
    input_path = "data/curated/features_daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError("features_daily.parquet not found. Run build_features.py first.")

    df = pd.read_parquet(input_path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # --- train/test split ---
    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()

    # --- params ---
    HOLD_DAYS = 3
    COST_BPS = 5
    COST = COST_BPS / 10_000

    # Mean reversion threshold: start simple
    # If 5-day log return is less than -2%, we consider it "oversold"
    THRESH = 0.02

    # Optional: avoid extreme volatility days (light filter)
    # We'll trade only when rvol_10 is <= the 80th percentile of the test set.
    VOL_PCTL = 0.80
    VOL_THRESHOLD = float(test["rvol_10"].quantile(VOL_PCTL))

    records = []

    for symbol, sdf in test.groupby("symbol"):
        sdf = sdf.copy().reset_index(drop=True)

        position = 0.0
        days_left = 0

        for i in range(len(sdf)):
            row = sdf.loc[i]
            trade_cost = 0.0

            vol_ok = row["rvol_10"] <= VOL_THRESHOLD
            oversold = row["ret_5d"] < -THRESH

            if days_left > 0:
                days_left -= 1

            # Entry
            if position == 0.0 and vol_ok and oversold:
                position = 1.0
                days_left = HOLD_DAYS
                trade_cost = position * COST

            # Exit
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
                    "oversold": oversold,
                    "vol_ok": vol_ok,
                }
            )

    bt = pd.DataFrame(records)

    daily = (
        bt.groupby("date", as_index=False)["net_ret"]
        .mean()
        .rename(columns={"net_ret": "portfolio_ret"})
    )

    daily["cum_log_ret"] = daily["portfolio_ret"].cumsum()
    daily["equity"] = np.exp(daily["cum_log_ret"])

    total_return = daily["equity"].iloc[-1] - 1
    avg_daily = daily["portfolio_ret"].mean()
    vol_daily = daily["portfolio_ret"].std()
    sharpe = (avg_daily / vol_daily) * np.sqrt(252) if vol_daily > 0 else np.nan

    print("Mean Reversion Backtest (single split)")
    print("-------------------------------------")
    print(f"Holding days:      {HOLD_DAYS}")
    print(f"Oversold thresh:   ret_5d < {-THRESH:.2%}")
    print(f"Vol filter:        rvol_10 <= {VOL_THRESHOLD:.6e} (pctl={VOL_PCTL:.2f})")
    print(f"Total return:      {total_return:.2%}")
    print(f"Avg daily return:  {avg_daily:.6e}")
    print(f"Daily vol:         {vol_daily:.6e}")
    print(f"Sharpe (approx):   {sharpe:.3f}")


if __name__ == "__main__":
    main()
