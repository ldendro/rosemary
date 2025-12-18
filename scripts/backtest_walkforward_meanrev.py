"""
Walk-forward evaluation for the rule-based mean reversion strategy.

Expanding window:
- Train period: all years before test_year (only used to define thresholds if needed)
- Test period: one calendar year
- Strategy is rule-based (no ML), so "training" is minimal.

Outputs:
- data/results/walkforward_meanrev_results.csv
"""

import os
import numpy as np
import pandas as pd


def compute_fold_metrics(daily: pd.DataFrame) -> dict:
    daily = daily.copy()
    daily["cum_log_ret"] = daily["portfolio_ret"].cumsum()
    daily["equity"] = np.exp(daily["cum_log_ret"])

    total_return = daily["equity"].iloc[-1] - 1
    avg_daily = daily["portfolio_ret"].mean()
    vol_daily = daily["portfolio_ret"].std()
    sharpe = (avg_daily / vol_daily) * np.sqrt(252) if vol_daily > 0 else np.nan

    daily["rolling_max"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["rolling_max"] - 1
    max_drawdown = daily["drawdown"].min()

    return {
        "total_return": float(total_return),
        "avg_daily": float(avg_daily),
        "daily_vol": float(vol_daily),
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "max_drawdown": float(max_drawdown),
        "n_days": int(len(daily)),
    }


def run_mean_reversion_on_test(
    test: pd.DataFrame,
    hold_days: int,
    cost_bps: int,
    oversold_thresh: float,
    vol_pctl: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    test = test.copy()

    cost = cost_bps / 10_000
    vol_threshold = float(test["rvol_10"].quantile(vol_pctl))

    records = []

    for symbol, sdf in test.groupby("symbol"):
        sdf = sdf.copy().reset_index(drop=True)

        position = 0.0
        days_left = 0

        for i in range(len(sdf)):
            row = sdf.loc[i]
            trade_cost = 0.0

            vol_ok = row["rvol_10"] <= vol_threshold
            oversold = row["ret_5d"] < -oversold_thresh

            if days_left > 0:
                days_left -= 1

            if position == 0.0 and vol_ok and oversold:
                position = 1.0
                days_left = hold_days
                trade_cost = position * cost
            elif position > 0.0 and days_left == 0:
                trade_cost = position * cost
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

    return bt, daily, vol_threshold


def main() -> None:
    input_path = "data/curated/features_daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError("features_daily.parquet not found. Run build_features.py first.")

    df = pd.read_parquet(input_path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    if len(years) < 3:
        raise RuntimeError("Not enough years of data for walk-forward evaluation.")

    HOLD_DAYS = 3
    COST_BPS = 5
    OVERSOLD_THRESH = 0.02   # ret_5d < -2%
    VOL_PCTL = 0.80          # keep most days, just avoid extreme vol

    results = []
    all_bt = []

    for test_year in years[1:]:
        test_df = df[df["year"] == test_year].copy()
        if len(test_df) < 50:
            continue

        bt, daily, vol_threshold = run_mean_reversion_on_test(
            test=test_df,
            hold_days=HOLD_DAYS,
            cost_bps=COST_BPS,
            oversold_thresh=OVERSOLD_THRESH,
            vol_pctl=VOL_PCTL,
        )

        metrics = compute_fold_metrics(daily)
        results.append(
            {
                "test_year": int(test_year),
                "hold_days": HOLD_DAYS,
                "oversold_thresh": OVERSOLD_THRESH,
                "vol_pctl": VOL_PCTL,
                "vol_threshold": float(vol_threshold),
                **metrics,
            }
        )

        bt["test_year"] = int(test_year)
        all_bt.append(bt)

    results_df = pd.DataFrame(results).sort_values("test_year").reset_index(drop=True)

    os.makedirs("data/results", exist_ok=True)
    out_csv = "data/results/walkforward_meanrev_results.csv"
    results_df.to_csv(out_csv, index=False)

    print("Walk-forward mean reversion results saved to:", out_csv)
    print(results_df)

    out_bt = "data/results/walkforward_meanrev_bt.parquet"
    if all_bt:
        pd.concat(all_bt, ignore_index=True).to_parquet(out_bt, index=False)
        print("Detailed per-day records saved to:", out_bt)


if __name__ == "__main__":
    main()
