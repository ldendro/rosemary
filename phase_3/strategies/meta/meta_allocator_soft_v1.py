import pandas as pd
import numpy as np

def sigmoid(x: pd.Series | float, k: float = 10.0) -> pd.Series | float:
    """
    Smooth mapping from (-inf, +inf) to (0, 1)
    k controls steepness (higher = more threshold-like)
    """
    return 1.0 / (1.0 + np.exp(-k * x))

def trend_score(mom_60: float, mom_20: float, k: float = 8.0) -> float:
    """
    Trend confidence increases smoothly with positive momentum.
    """
    s60 = sigmoid(mom_60, k=k)
    s20 = sigmoid(mom_20, k=k)
    return float(s60 * s20)

def meanrev_score(mom_20: float, drawdown_60: float, vol_20: float, k: float = 8.0) -> float:
    """
    Mean-reversion confidence increases with negative mom_20, depper drawdowns, and lower volatility.
    """
    s_mom = sigmoid(-mom_20, k=k)
    s_dd = sigmoid(-drawdown_60, k=k)

    vol_gate = sigmoid(0.40 - vol_20, k=12.0)
    
    return float(s_mom * s_dd * vol_gate)

def cash_score(trend_s: float, meanrev_s: float, vol_20: float) -> float:
    """
    Cash gets an explicit score, its higher when both signals are weak or when volatility is high.
    """
    weak_signal = 1.0 - max(trend_s, meanrev_s)
    vol_risk = sigmoid(vol_20 - 0.35, k = 10.0)
    return float(0.7 * weak_signal + 0.3 * vol_risk)

def scores_to_weights(trend_s: float, meanrev_s: float, cash_s: float, eps: float = 1e-12) -> dict:
    """
    Normalize scores into weights that sum to 1
    """
    total = trend_s + meanrev_s + cash_s + eps
    return {
        "w_trend": trend_s / total,
        "w_meanrev": meanrev_s / total,
        "w_cash": cash_s / total
    }

def compute_soft_weights_row(row: pd.Series) -> dict:
    """
    Compute (w_trend, w_meanrev, w_cash) for one date row.
    Returns cash-only for early rows with NaNs.
    """
    mom_60 = row["mom_60"]
    mom_20 = row["mom_20"]
    vol_20 = row["vol_20"]
    dd_60  = row["drawdown_60"]

    if pd.isna(mom_60) or pd.isna(mom_20) or pd.isna(vol_20) or pd.isna(dd_60):
        return {"w_trend": 0.0, "w_meanrev": 0.0, "w_cash": 1.0}

    t = trend_score(mom_60, mom_20)
    m = meanrev_score(mom_20, dd_60, vol_20)
    c = cash_score(t, m, vol_20)

    return scores_to_weights(t, m, c)

def power_normalize_weights(w_trend: float, w_meanrev: float, w_cash: float, gamma: float, eps: float = 1e-12) -> dict:
    """"
    Apply power nomralization to soft weights to restore convexity.
    gamma = 1.0 -> no change (pure soft)
    gamma > 1.0 -> more concentrated (commitment)
    gamme -> inf -> hard-like behavior
    """
    wt = max(w_trend, 0.0) ** gamma 
    wm = max(w_meanrev, 0.0) ** gamma
    wc = max(w_cash, 0.0) ** gamma

    total = wt + wm + wc + eps
    return {
        "w_trend": wt / total,
        "w_meanrev": wm / total,
        "w_cash": wc / total
    }

def build_soft_meta_returns(trend_out: pd.DataFrame, meanrev_out: pd.DataFrame, regime_df: pd.DataFrame, gamma: float = 1.0) -> pd.DataFrame:
    """
    Produces:
      date, w_trend, w_meanrev, w_cash, meta_raw_ret
    """
    merged = (
        regime_df[["date", "mom_20", "mom_60", "vol_20", "drawdown_60"]]
        .merge(
            trend_out[["date", "raw_ret"]].rename(columns={"raw_ret": "trend_raw_ret"}),
            on="date",
            how="inner",
        )
        .merge(
            meanrev_out[["date", "raw_ret"]].rename(columns={"raw_ret": "meanrev_raw_ret"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    weights = merged.apply(compute_soft_weights_row, axis=1, result_type="expand")

    weights = weights.apply(
        lambda r: power_normalize_weights(
            r["w_trend"], r["w_meanrev"], r["w_cash"], gamma
        ),
        axis=1,
        result_type="expand",
    )

    merged = pd.concat([merged, weights], axis=1)


    merged["meta_raw_ret"] = (
        merged["w_trend"] * merged["trend_raw_ret"] +
        merged["w_meanrev"] * merged["meanrev_raw_ret"]
    )

    return merged[["date", "w_trend", "w_meanrev", "w_cash", "meta_raw_ret"]]
