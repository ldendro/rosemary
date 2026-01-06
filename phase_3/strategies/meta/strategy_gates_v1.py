import pandas as pd

DEFAULT_GATE_PARAMS = {
    "max_vol_20_global": 0.60,
    "trend_min_mom60": 0.00,
    "mr_max_abs_mom60": 0.10
}

def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def build_strategy_gates_v1(regime_df: pd.DataFrame, p: dict = DEFAULT_GATE_PARAMS) -> pd.DataFrame:
    """
    Build conservative strategy availability gates.
    
    Expects regime_df columns:
        - date
        - mom_60
        - vol_20
    
    Returns DataFrame with:
        - date
        - trend_allowed (bool)
        - meanrev_allowed (bool)
    """
    _require_columns(regime_df, ["date", "mom_60", "vol_20"])
    df = regime_df[["date", "mom_60", "vol_20"]].copy()

    safe_env = df["vol_20"] < p["max_vol_20_global"]
    safe_env = safe_env.fillna(False)

    trend_allowed = safe_env & (df["mom_60"] > p["trend_min_mom60"])
    trend_allowed = trend_allowed.fillna(False)

    mr_allowed = safe_env & (df["mom_60"].abs() < p["mr_max_abs_mom60"])
    mr_allowed = mr_allowed.fillna(False)

    out = pd.DataFrame({
        "date": df["date"],
        "trend_allowed": trend_allowed.astype(bool),
        "meanrev_allowed": mr_allowed.astype(bool),
        })
    
    return out