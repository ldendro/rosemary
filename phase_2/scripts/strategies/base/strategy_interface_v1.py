import pandas as pd
import numpy as np

def compute_close_to_close_returns(df: pd.DataFrame, price_col: str = "adj_close") -> pd.Series:
    """
    Canonical daily return definition for Phase 2
    ret[t] = price[t] / price[t-1] - 1
    """
    return df[price_col].pct_change()

def apply_transaction_costs(position: pd.Series, cost_per_side_bps: float) -> pd.Series:
    """
    Simple transaction cost model.

    Turnover[t] = |position[t] - position[t-1]|
    Cost[t] = turnover[t] * cost_per_side_bps
    """
    turnover = position.diff().abs().fillna(0.0)
    cost = turnover * (cost_per_side_bps / 10_000.0)
    return cost

def finalize_strategy_output(df: pd.DataFrame, strategy_name: str, position_col: str = "position", price_col: str = "adj_close", cost_per_side_bps: float = 5.0, ) -> pd.DataFrame:
    """
    Finalize the strategy output by calculating returns and applying transaction costs.

    Parameters:
    - df: DataFrame containing the strategy data.
    - strategy_name: Name of the strategy.
    - position_col: Column name for the position series.
    - price_col: Column name for the price series.
    - cost_per_side_bps: Transaction cost per side in basis points.

    Returns:
    - DataFrame with the strategy output including returns and transaction costs.
    """
    required = {"date", price_col, position_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["position"] = out[position_col].astype(float).clip(0.0, 1.0)
    out["ret_1d"] = compute_close_to_close_returns(out, price_col)
    pos_applied = out["position"].shift(1).fillna(0.0)
    gross_ret = pos_applied * out["ret_1d"].fillna(0.0)
    costs = apply_transaction_costs(out["position"], cost_per_side_bps)
    raw_ret = gross_ret - costs

    return pd.DataFrame({
        "date": out["date"],
        "position": out["position"],
        "raw_ret": raw_ret,
        "strategy_name": strategy_name,
    })