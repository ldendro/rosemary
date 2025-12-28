import pandas as pd
import numpy as np

TREND = "TREND"
MEANREV = "MEANREV"
CASH = "CASH"

def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds simple regime features for the meta allocator.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["ret_1d"] = out["adj_close"].pct_change()

    out["mom_20"] = out["adj_close"].pct_change(20)
    out["mom_60"] = out["adj_close"].pct_change(60)

    out["vol_20"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)

    roll_max_60 = out["adj_close"].rolling(60).max()
    out["drawdown_60"] = out["adj_close"] / roll_max_60 - 1.0

    return out

def choose_state_v1(row: pd.Series) -> str:
    """
    Simple, interpretable gating rule (v1):

    - TREND when medium-term momentum is positive and short-term momentum isn't too negative
    - MEANREV when short-term momentum is negative and we are in drawdown but volatility isn't insane
    - otherwise CASH
    """
    mom_60 = row["mom_60"]
    mom_20 = row["mom_20"]
    vol_20 = row["vol_20"]
    dd_60 = row["drawdown_60"]

    if pd.isna(mom_60) or pd.isna(mom_20) or pd.isna(vol_20) or pd.isna(dd_60):
        return CASH

    if (mom_60 > 0.0) and (mom_20 > -0.01):
        return TREND
    
    if (mom_20 < -0.02) and (dd_60 < -0.03) and (vol_20 < 0.40):
        return MEANREV

    return CASH

def build_meta_raw_returns(trend_out: pd.DataFrame, meanrev_out: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces:
      date, state, meta_raw_ret
    """
    merged = regime_df[["date", "mom_20", "mom_60", "vol_20", "drawdown_60"]].merge(
        trend_out[["date", "raw_ret"]].rename(columns={"raw_ret": "trend_raw_ret"}),
        on="date",
        how="inner",
    ).merge(
        meanrev_out[["date", "raw_ret"]].rename(columns={"raw_ret": "meanrev_raw_ret"}),
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    merged["state"] = merged.apply(choose_state_v1, axis=1)

    merged["meta_raw_ret"] = 0.0
    merged.loc[merged["state"] == TREND, "meta_raw_ret"] = merged.loc[merged["state"] == TREND, "trend_raw_ret"]
    merged.loc[merged["state"] == MEANREV, "meta_raw_ret"] = merged.loc[merged["state"] == MEANREV, "meanrev_raw_ret"]

    return merged[["date", "state", "meta_raw_ret"]]
