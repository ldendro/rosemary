from typing import Dict, List
import pandas as pd
import numpy as np

SECTOR_MAP: Dict[str, str] = {
    "SPY": "Broad",
    "QQQ": "BroadTech",
    "IWM": "SmallCap",

    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",

    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLP": "Staples",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
}

def apply_per_asset_cap(w: pd.DataFrame, max_weight: float = 0.20) -> pd.DataFrame:
    """
    Apply a hard cap to each asset's weight at each timestamp.

    Parameters
    ----------
    w : pd.DataFrame
        Rows = dates, columns = assets, values = weights
    max_weight : float
        Maximum allowed weight per asset (e.g. 0.20 for 20%)

    Returns
    -------
    pd.DataFrame
        Weight DataFrame after clipping + row-wise normalization
    """
    w_clipped = w.clip(upper=max_weight)
    row_sums = w_clipped.sum(axis=1)
    nonzero = row_sums != 0
    w_norm = w_clipped.copy()
    w_norm.loc[nonzero] = w_clipped.loc[nonzero].div(row_sums[nonzero], axis=0)
    return w_norm

def apply_sector_cap(w: pd.DataFrame, sector_map: Dict[str,str], max_sector_weight: float = 0.40) -> pd.DataFrame:
    """
    Apply a hard cap to sector-level weights.

    Parameters
    ----------
    w: pd.DataFrame
        Rows = dates, columns = assets, values = weights (sum ~ 1.0)
    sector_map: Dict[str,str]
        Mapping ticker -> sector name
    max_sector_weight : float
        Maximum allowed total weight per sector (e.g. 0.40 for 40%)

    Returns 
    -------
    pd.DataFrame
        Sector-capped and row-wise normalized weight DataFrame
    """

    w_adj = w.copy()

    assets = w.columns
    sectors = pd.Series({a: sector_map[a] for a in assets})

    unique_sectors = sectors.unique()

    for date_idx in w.index:
        row = w_adj.loc[date_idx]

        sector_totals = {}
        for sec in unique_sectors:
            sector_totals[sec] = row[sectors[sectors == sec].index].sum()

        violators = [sec for sec, tot in sector_totals.items() if tot > max_sector_weight]

        if not violators:
            continue

        row_adj = row.copy()
        for sec in violators:
            sec_assets = sectors[sectors == sec].index
            tot = sector_totals[sec]
            scale = max_sector_weight / tot
            row_adj.loc[sec_assets] *= scale

        s = row_adj.sum()
        if s > 0:
            row_adj = row_adj / s

        w_adj.loc[date_idx] = row_adj

    return w_adj

def apply_min_diversification(w: pd.DataFrame, min_assets: int = 6) -> pd.DataFrame:
    """
    Ensure each row holds at least 'min_assets' non-zero weights.
    If fewer than min_assets are non-zero, we will:
        1) Identify zero-weight assets
        2) Activate assets with smallest zero weights equally 
        3) Renormalize the entire row

    This avoids pathological concentration after caps
    """
    
    w_adj = w.copy()

    for date_idx in w.index:
        row = w_adj.loc[date_idx]

        nonzero_mask = row > 0
        nonzero_count = nonzero_mask.sum()

        if nonzero_count >= min_assets:
            continue

        needed = min_assets - nonzero_count

        zero_assets = row[~nonzero_mask].index.tolist()

        if len(zero_assets) == 0:
            continue

        activate_assets = zero_assets[:needed]

        placeholder = 1e-6
        row_adj = row.copy()
        for a in activate_assets:
            row_adj[a] = placeholder 

        s = row_adj.sum()
        if s > 0:
            row_adj = row_adj / s

        w_adj.loc[date_idx] = row_adj

    return w_adj

def apply_all_risk_caps(w: pd.DataFrame, sector_map: Dict[str, str] = SECTOR_MAP, max_weight_per_asset: float = 0.20, max_sector_weight: float = 0.40, min_assets_held: int = 6) -> pd.DataFrame:
    """
    Apply all Phase 5A risk caps in sequence.

    Order of operations:
      1) Per-asset hard cap
      2) Sector hard cap
      3) Min diversification
      4) Final normalization (safety)

    Returns
    -------
    pd.DataFrame
        Adjusted weight DataFrame
    """

    w1 = apply_per_asset_cap(w, max_weight=max_weight_per_asset)

    w2 = apply_sector_cap(w1, sector_map, max_sector_weight=max_sector_weight)

    w3 = apply_min_diversification(w2, min_assets=min_assets_held)

    row_sums = w3.sum(axis=1)
    nonzero = row_sums != 0
    w_final = w3.copy()
    w_final.loc[nonzero] = w3.loc[nonzero].div(row_sums[nonzero], axis=0)

    return w_final
