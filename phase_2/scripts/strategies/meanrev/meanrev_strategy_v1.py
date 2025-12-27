import pandas as pd
import numpy as np

from ..base.strategy_interface_v1 import finalize_strategy_output

def build_meanrev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features needed for mean reversion strategy.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["ret_1d"] = out["adj_close"].pct_change()
    out["ret_5d"] = out["adj_close"].pct_change(5)

    out["vol_20"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)

    return out

def generate_meanrev_positions(df: pd.DataFrame, entry_ret_5d_threshold: float = -0.02, hold_days: int = 3, max_vol_annual: float = 0.40) -> pd.Series:
    """
    Generate long/cash positions for mean reversion.
    """

    feat = build_meanrev_features(df)

    can_enter = (feat["ret_5d"] < entry_ret_5d_threshold) & (feat["vol_20"] < max_vol_annual)

    can_enter = can_enter.fillna(False)

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

def run_meanrev_strategy_v1(df: pd.DataFrame, entry_ret_5d_threshold: float = -0.02, hold_days: int = 3, max_vol_annual: float = 0.40, cost_per_side_bps: float = 5.0) -> pd.DataFrame:
    """
    Run mean reversion strategy (Phase 2 v1).
    """
    working = df.copy()

    working["position"] = generate_meanrev_positions(working,entry_ret_5d_threshold=entry_ret_5d_threshold,hold_days=hold_days,max_vol_annual=max_vol_annual).values

    out = finalize_strategy_output(working,strategy_name="meanrev_v1",position_col="position",price_col="adj_close",cost_per_side_bps=cost_per_side_bps)

    return out
