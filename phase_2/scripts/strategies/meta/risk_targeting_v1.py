import pandas as pd
import numpy as np

def apply_vol_targeting(raw_ret: pd.Series, target_vol_annual: float = 0.10, lookback: int = 20, max_leverage: float = 1.0) -> pd.DataFrame:
    """
    Vol targeting: 
    lev[t] = target_daily_vol / realized_daily_vol

    Returns a DataFrame with:
        lev, meta_ret
    """

    df = pd.DataFrame({"raw_ret": raw_ret}).copy()

    target_daily = target_vol_annual / np.sqrt(252)
    realized_daily = df["raw_ret"].rolling(lookback).std()

    lev = target_daily / realized_daily
    lev = lev.clip(lower=0.0, upper=max_leverage).fillna(0.0)

    df["lev"] = lev
    df["meta_ret"] = df["raw_ret"] * df["lev"]
    return df