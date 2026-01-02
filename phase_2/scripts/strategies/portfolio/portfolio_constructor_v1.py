import pandas as pd
import numpy as np

def align_asset_returns(asset_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert dict of per-asset daily DataFrames into a single wide DataFrame:

    index = date
    columns = assets
    values = meta_raw_ret

    Expects each DataFrame to have:
    - date
    - asset
    - meta_raw_ret
    """
    frames = []
    for asset, df in asset_daily.items():
        tmp = df[["date", "meta_raw_ret"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.set_index("date")
        tmp = tmp.rename(columns={"meta_raw_ret": asset})
        frames.append(tmp)

    wide = pd.concat(frames, axis=1).sort_index()

    return wide

def compute_inverse_vol_weights(ret_wide: pd.DataFrame, lookback: int = 20, max_weight: float = 0.7) -> pd.DataFrame:
    """
    Rolling inverse-vol weights.
    
    For each date:
        vol_i = std(ret_i over lookback)
        inv_i = 1 / vol_i
        w_i = inv_i / sum(inv_i)
         
    Notes:
    - weights are based on meta_raw_ret (pre vol targeting)
    - weights are capped to avoid concentration      
    """
    vol = ret_wide.rolling(lookback).std()
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    if max_weight is not None:
        w = w.clip(upper=max_weight)
        w = w.div(w.sum(axis=1), axis=0)    

    w = w.fillna(0.0)

    return w

def build_portfolio_raw_returns(ret_wide: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """
    portfolio_raw_ret[t] = sum_i weights[t, i] * ret_wide[t, i]
    """
    common_cols = [c for c in ret_wide.columns if c in weights.columns]
    port = (weights[common_cols] * ret_wide[common_cols]).sum(axis=1)
    port.name = "portfolio_raw_ret"
    return port 