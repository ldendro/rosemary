from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import cvxpy as cp

from phase_4.scripts.utils.covariance_estimators_v1 import sample_covariance

def solve_min_variance_weights(ret_wide: pd.DataFrame, lookback: int = 60, sector_map: Optional[Dict[str,str]] = None, max_weight_per_asset: float = 0.20, max_sector_weight: float = 0.40) -> pd.Series:
    """
    Solve for min-variance long-only weights with per-asset and sector caps.
    
    Parameters
    ----------
    ret_wide : pd.DataFrame
        Rows = dates, columns = assets, values = daily meta returns.
    lookback : int
        Number of most recent days to use for covariance estimations.
    sector_map : dict, optional
        Mapping ticker -> sector. If None, sector caps are skipped.
    max_weight_per_asset : float
        Hard per-asset cap (e.g. 0.20).
    max_Sector_weight : float
        Hard sector cap (e.g. 0.40).

    Returns
    -------
    pd.Series
        Optimal weights indexed by asset (sum ~ 1.0)
        If problem is infeasible or covariance is empty, returns equal weight
    """
    assets = list(ret_wide.columns)

    window = ret_wide.dropna().iloc[-lookback:]

    cov = sample_covariance(window, min_periods=lookback, assets=assets)

    if cov.empty or cov.isna().values.any():
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)
    
    n = len(assets)
    w = cp.Variable(n)

    Sigma = cov.values
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = []
    constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1.0)

    if max_weight_per_asset is not None:
        constraints.append(w <= max_weight_per_asset)

    if sector_map is not None:
        sector_to_indices: Dict[str, List[int]] = {}
        for idx, a in enumerate(assets):
            sec = sector_map.get(a, "Unknown")
            sector_to_indices.setdefault(sec, []).append(idx)

        for sec, idx_list in sector_to_indices.items():
            if max_sector_weight is not None:
                constraints.append(cp.sum(w[idx_list]) <= max_sector_weight)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.ECOS)
    except Exception as e:
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)

    if w.value is None:
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)

    w_opt = np.array(w.value).flatten()
    w_opt = np.clip(w_opt, 0.0, None)

    s = w_opt.sum()
    if s > 0:
        w_opt = w_opt / s
    else:
        w_opt = np.ones(n) / n

    return pd.Series(w_opt, index=assets) 
