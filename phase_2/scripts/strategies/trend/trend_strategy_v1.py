import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from ..base.strategy_interface_v1 import finalize_strategy_output

def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build trend-following features on canonical SPY data. 
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["ret_1d"] = out["adj_close"].pct_change()
    out["ret_5d"] = out["adj_close"].pct_change(5)
    out["ret_10d"] = out["adj_close"].pct_change(10)
    out["ret_20d"] = out["adj_close"].pct_change(20)

    vol_ma_10 = out["volume"].rolling(window=10).mean()
    out["rvol_10"] = out["volume"] / vol_ma_10

    return out

def build_forward_return_label(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """
    Forward return label:
    label[t] = adj_close[t + horizon] / adj_close[t] - 1
    """
    label = df["adj_close"].pct_change(horizon_days).shift(-horizon_days)
    return label

def fit_trend_model(train_df: pd.DataFrame, horizon_days: int = 5) -> LinearRegression:
    """
    Fit a linear regression on training data only
    """
    feat = build_trend_features(train_df)
    feat["label"] = build_forward_return_label(feat, horizon_days)

    feature_cols = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "rvol_10"]

    usable = feat.dropna(subset=feature_cols + ["label"]).copy()

    if len(usable) < 200:
        raise ValueError("Not enough data to fit trend model.")
    
    X = usable[feature_cols].values
    y = usable["label"].values

    model = LinearRegression()
    model.fit(X, y)

    return model

def generate_trend_positions(df: pd.DataFrame, model: LinearRegression, prediction_threshold: float = 0.0, hold_days: int = 5) -> pd.Series:
    """ 
    Generate long/cash positions using a fixed holding period
    """
    feat = build_trend_features(df)
    feat = feat.reset_index(drop=True)
    feature_cols = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "rvol_10"]

    preds = pd.Series(np.nan, index=feat.index)

    ready = feat[feature_cols].notna().all(axis=1)
    preds.loc[ready] = model.predict(feat.loc[ready, feature_cols].values)

    can_enter = (
        (preds > prediction_threshold) & (feat["ret_20d"] > 0)
        )
    
    pos = np.zeros(len(feat))
    hold_remaining = 0

    for i in range(len(feat)):
        if hold_remaining > 0:
            pos[i] = 1.0
            hold_remaining -= 1
            continue

        if bool(can_enter.iloc[i]):
            pos[i] = 1.0
            hold_remaining = hold_days - 1
        else:
            pos[i] = 0.0

    return pd.Series(pos, index=feat.index, name="position")

def run_trend_strategy_v1(df: pd.DataFrame, train_df: pd.DataFrame, prediction_threshold: float = 0.0, hold_days: int = 5, cost_per_side_bps: float = 5.0) -> pd.DataFrame:
    """
    Run trend strategy on df using a model fit on train_df.
    """
    model = fit_trend_model(train_df)

    working = df.copy()
    working["position"] = generate_trend_positions(working, model, prediction_threshold=prediction_threshold, hold_days=hold_days).values

    out = finalize_strategy_output(working, strategy_name="trend_v1", position_col="position", price_col="adj_close", cost_per_side_bps=cost_per_side_bps)

    return out
