"""
Train a baseline predictive model for the Rosemary project.

This script trains a simple linear regression model to predict next-day
returns using a minimal feature set. The goal is NOT performance, but to
validate that the feature pipeline contains usable signal.

Inputs:
- data/curated/features_daily.parquet

Features:
- ret_1d
- rvol_10

Target:
- y_ret_1d
"""

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main() -> None:
    input_path = "data/curated/features_daily.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "features_daily.parquet not found. Run build_features.py first."
        )

    df = pd.read_parquet(input_path)

    FEATURES = ["ret_1d", "rvol_10"]
    TARGET = "y_ret_1d"

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Baseline Linear Regression Results")
    print("---------------------------------")
    print(f"MSE: {mse:.6e}")
    print(f"R²:  {r2:.6f}")

    print("\nModel coefficients:")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name}: {coef:.6e}")

    print(f"\nIntercept: {model.intercept_:.6e}")


if __name__ == "__main__":
    main()
