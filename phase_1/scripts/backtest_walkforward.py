"""
Walk-forward backtest for Rosemary.

Runs an expanding-window walk-forward evaluation:
- Train on all data up to a cutoff year
- Test on the next calendar year
- Repeat forward

Strategy logic (fixed):
- LinearRegression model
- Prediction threshold: 0.5 * pred_std (computed on that fold's test set)
- Holding period: 5 days
- Trend filter: ret_20d > 0
- Transaction costs: 5 bps on entry and exit

Outputs:
- data/results/walkforward_results.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def compute_fold_metrics(daily: pd.DataFrame) -> dict:
    """Compute simple performance metrics for a daily return series."""
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
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,  # handle nan
        "max_drawdown": float(max_drawdown),
        "n_days": int(len(daily)),
    }


def run_strategy_on_test(
    test: pd.DataFrame,
    features: list[str],
    model: LinearRegression,
    hold_days: int,
    cost_bps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply holding + trend filter strategy to a test set.
    Returns:
      - bt: per-symbol daily records (position, net_ret)
      - daily: portfolio aggregated daily returns
    """
    test = test.copy()
    test["prediction"] = model.predict(test[features])

    pred_std = test["prediction"].std()
    threshold = 0.5 * pred_std

    cost = cost_bps / 10_000

    records = []

    for symbol, sdf in test.groupby("symbol"):
        sdf = sdf.copy().reset_index(drop=True)

        position = 0.0
        days_left = 0

        for i in range(len(sdf)):
            row = sdf.loc[i]
            trade_cost = 0.0

            trend_ok = row["ret_20d"] > 0

            if days_left > 0:
                days_left -= 1

            # Entry
            if position == 0.0 and trend_ok and row["prediction"] > threshold:
                # sizing
                position = min(row["prediction"] / pred_std, 1.0) if pred_std > 0 else 0.0
                days_left = hold_days
                trade_cost = position * cost

            # Exit
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
                    "prediction": row["prediction"],
                    "threshold": threshold,
                    "trend_ok": trend_ok,
                    "net_ret": net_ret,
                }
            )

    bt = pd.DataFrame(records)

    daily = (
        bt.groupby("date", as_index=False)["net_ret"]
        .mean()
        .rename(columns={"net_ret": "portfolio_ret"})
    )

    return bt, daily


def main() -> None:
    input_path = "data/curated/features_daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "features_daily.parquet not found. Run build_features.py first."
        )

    df = pd.read_parquet(input_path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    FEATURES = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "rvol_10"]
    TARGET = "y_ret_1d"

    HOLD_DAYS = 5
    COST_BPS = 5

    years = sorted(df["year"].unique())
    if len(years) < 3:
        raise RuntimeError("Not enough years of data for walk-forward evaluation.")

    # We'll evaluate year-by-year. Need at least one prior year for training.
    results = []
    all_bt = []

    for test_year in years[1:]:
        train_df = df[df["year"] < test_year]
        test_df = df[df["year"] == test_year].copy()

        # Guard: if test year has too little data, skip
        if len(test_df) < 50 or len(train_df) < 200:
            continue

        model = LinearRegression()
        model.fit(train_df[FEATURES], train_df[TARGET])

        bt, daily = run_strategy_on_test(
            test=test_df,
            features=FEATURES,
            model=model,
            hold_days=HOLD_DAYS,
            cost_bps=COST_BPS,
        )

        metrics = compute_fold_metrics(daily)

        results.append(
            {
                "test_year": int(test_year),
                "train_start": int(train_df["year"].min()),
                "train_end": int(train_df["year"].max()),
                **metrics,
            }
        )

        bt["test_year"] = int(test_year)
        all_bt.append(bt)

    results_df = pd.DataFrame(results).sort_values("test_year").reset_index(drop=True)

    os.makedirs("data/results", exist_ok=True)
    results_path = "data/results/walkforward_results.csv"
    results_df.to_csv(results_path, index=False)

    print("Walk-forward results saved to:", results_path)
    print(results_df)

    # Optional detailed artifact (can be large)
    bt_path = "data/results/walkforward_bt.parquet"
    if all_bt:
        pd.concat(all_bt, ignore_index=True).to_parquet(bt_path, index=False)
        print("Detailed per-day records saved to:", bt_path)


if __name__ == "__main__":
    main()
