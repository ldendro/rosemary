from typing import Optional, List
import pandas as pd
import numpy as np

def sample_covariance(returns: pd.DataFrame, min_periods: int = 60, assets: Optional[List[str]] = None) -> pd.DataFrame:
    """
    COmpute a simple sample covariance matrix from asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows = dates, columns = asset tickers, values = daily returns.
    min_periods : int
        Minimum number of non-NaN observations required to compute covariance.
    assets : list of str, optional
        Optional subset of columns to include. If None, use all columns.
    """
    if assets is None:
        assets = list(returns.columns)

    sub = returns[assets].dropna(how="all")

    if sub.shape[0] < min_periods:
        return pd.DataFrame(index=assets, columns=assets, dtype=float)
    
    cov = sub.cov()

    cov = cov.reindex(index=assets, columns=assets)

    return cov