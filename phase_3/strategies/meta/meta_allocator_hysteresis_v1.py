import pandas as pd 

TREND = "TREND"
MEANREV = "MEANREV"
CASH = "CASH"

DEFAULT_HYST_PARAMS = {
    "trend_enter_mom60": 0.00,
    "trend_enter_mom20": -0.005,   
    "trend_exit_mom60": -0.02,     
    "trend_exit_mom20": -0.02,

    "mr_enter_mom20": -0.025,
    "mr_enter_dd60": -0.03,
    "mr_enter_vol20_max": 0.40,

    "mr_exit_mom20": -0.005,       
    "mr_exit_dd60": -0.01,         
    "mr_exit_vol20_max": 0.45,     
}

def decide_state_hysteresis_v1(row: pd.Series, prev_state: str, p: dict = DEFAULT_HYST_PARAMS) -> str:
    """
    Hysteresis state machine:
      - separate enter/exit rules
      - prev_state matters
    """

    mom_60 = row["mom_60"]
    mom_20 = row["mom_20"]
    vol_20 = row["vol_20"]
    dd_60  = row["drawdown_60"]

    if pd.isna(mom_60) or pd.isna(mom_20) or pd.isna(vol_20) or pd.isna(dd_60):
        return CASH

    trend_enter = (mom_60 > p["trend_enter_mom60"]) and (mom_20 > p["trend_enter_mom20"])
    trend_exit  = (mom_60 < p["trend_exit_mom60"]) or  (mom_20 < p["trend_exit_mom20"])

    mr_enter = (
        (mom_20 < p["mr_enter_mom20"]) and
        (dd_60 < p["mr_enter_dd60"]) and
        (vol_20 < p["mr_enter_vol20_max"])
    )

    mr_exit = (
        (mom_20 > p["mr_exit_mom20"]) or
        (dd_60 > p["mr_exit_dd60"]) or
        (vol_20 > p["mr_exit_vol20_max"])
    )

    
    if prev_state == TREND:
        if trend_exit:
            if mr_enter:
                return MEANREV
            return CASH
        return TREND

    if prev_state == MEANREV:
        if mr_exit:
            if trend_enter:
                return TREND
            return CASH
        return MEANREV

    if trend_enter:
        return TREND
    if mr_enter:
        return MEANREV
    return CASH

def build_state_series_hysteresis_v1(regime_df: pd.DataFrame, p: dict = DEFAULT_HYST_PARAMS) -> pd.Series:
    """
    Build a daily state series using hysteresis.
    regime_df must be sorted by date and contain:
      date, mom_20, mom_60, vol_20, drawdown_60
    """
    df = regime_df.sort_values("date").reset_index(drop=True)

    states = []
    prev = CASH

    for _, row in df.iterrows():
        nxt = decide_state_hysteresis_v1(row, prev_state=prev, p=p)
        states.append(nxt)
        prev = nxt

    return pd.Series(states, index=df.index, name="state")

def build_meta_raw_returns_hysteresis_v1(trend_out: pd.DataFrame, meanrev_out: pd.DataFrame, regime_df: pd.DataFrame, p: dict = DEFAULT_HYST_PARAMS) -> pd.DataFrame:
    """
    Returns:
      date, state, meta_raw_ret
    """
    merged = (
        regime_df[["date", "mom_20", "mom_60", "vol_20", "drawdown_60"]]
        .merge(trend_out[["date", "raw_ret"]].rename(columns={"raw_ret": "trend_raw_ret"}), on="date", how="inner")
        .merge(meanrev_out[["date", "raw_ret"]].rename(columns={"raw_ret": "meanrev_raw_ret"}), on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    merged["state"] = build_state_series_hysteresis_v1(merged, p=p)

    merged["meta_raw_ret"] = 0.0
    merged.loc[merged["state"] == TREND, "meta_raw_ret"] = merged.loc[merged["state"] == TREND, "trend_raw_ret"]
    merged.loc[merged["state"] == MEANREV, "meta_raw_ret"] = merged.loc[merged["state"] == MEANREV, "meanrev_raw_ret"]

    return merged[["date", "state", "meta_raw_ret"]]

