"""
Data Transformations Module
===========================

Functions for transforming financial data and extracting factors via PCA.

Outlier Treatment:
- All series are winsorized at the 1st and 99th percentiles after transformation
- This is a standard robust procedure in finance (see Welch & Goyal, 2008)
- No data is excluded, only extreme values are capped
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List

# Séries à EXCLURE de la PCA selon Andreou et al. (2013)
# Le papier exclut uniquement la variable cible et les séries non-daily
EXCLUDE_FROM_PCA = [
    "GDP CQOQ Index",     # Variable cible (quarterly)
    "NFP TCH Index",      # Fréquence mensuelle (non-daily)
]

# IMPORTANT: Contrairement à mon approche précédente, le papier Andreou et al. (2013)
# utilise TOUTES les séries daily, même avec outliers, en appliquant:
# 1. Winsorization aux 1er/99e percentiles 
# 2. Standardization avant PCA
# 3. Aucune exclusion arbitraire de séries

# Percentiles pour la winsorization (standard en finance quantitative)
WINSOR_LOWER = 0.01  # 1st percentile
WINSOR_UPPER = 0.99  # 99th percentile


def winsorize_series(s: pd.Series, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER) -> pd.Series:
    """
    Winsorize a series at specified percentiles.
    
    This is a standard robust procedure that caps extreme values without
    excluding them, preserving sample size and avoiding selection bias.
    
    Reference: Standard practice in empirical finance (Welch & Goyal, 2008)
    """
    if s.notna().sum() < 30:
        return s
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(q_low, q_high)


def auto_transform_series(
    s: pd.Series, 
    scale_ret: float = 100.0, 
    scale_diff: float = 100.0,
    winsorize: bool = True
) -> pd.Series:
    """
    Automatically transform a financial series based on its characteristics.
    
    Transformation rules:
    - Price/index series (strictly positive): log-returns
    - Rate/spread series (can be negative or zero): first differences
    
    All series are winsorized at 1st/99th percentiles after transformation.
    This is academically rigorous and does NOT exclude any observations.
    """
    s = s.astype(float)

    if s.notna().sum() < 30:
        return s * np.nan

    # Standard transformation based on data characteristics
    if (s.dropna() > 0).all():
        # All positive values -> log-returns (standard for prices)
        transformed = (np.log(s) - np.log(s.shift(1))) * scale_ret
    else:
        # Contains zero or negative values -> first differences
        transformed = (s - s.shift(1)) * scale_diff
    
    # Winsorize to limit outlier impact (standard robust procedure)
    if winsorize:
        transformed = winsorize_series(transformed)
    
    return transformed


def transform_panel_auto(
    df_daily: pd.DataFrame, 
    target_col: str = "GDP CQOQ Index",
    exclude_cols: List[str] = None,
    winsorize: bool = True
) -> pd.DataFrame:
    """
    Apply automatic transformations to all columns of a daily panel.
    
    The target column (GDP) is kept in levels, all other columns are transformed
    to stationary series using log-returns or first differences.
    
    All transformed series are winsorized at 1%/99% percentiles.
    
    Parameters:
    -----------
    df_daily : DataFrame
        Raw daily data panel
    target_col : str
        Name of the target variable (kept in levels)
    exclude_cols : List[str]
        Additional columns to exclude from transformation
    winsorize : bool
        Whether to apply winsorization (default: True)
    """
    exclude_cols = exclude_cols or []
    out = {}
    
    for col in df_daily.columns:
        if col == target_col:
            # Keep target in levels (already a growth rate)
            out[col] = df_daily[col].astype(float)
        elif col in exclude_cols:
            # Explicitly excluded columns
            out[col] = df_daily[col].astype(float) * np.nan
        else:
            # Standard transformation with winsorization
            out[col] = auto_transform_series(df_daily[col], winsorize=winsorize)
    
    return pd.DataFrame(out, index=df_daily.index)


def build_quarterly_target(df_daily: pd.DataFrame, target_col: str = "GDP CQOQ Index") -> pd.Series:
    """
    Extract quarterly target variable from daily panel.
    Takes the last observation of each quarter for the target variable.
    """
    s = df_daily[target_col].dropna().sort_index()
    yq = s.groupby(s.index.to_period("Q")).last()
    yq.index = yq.index.to_timestamp("Q")
    return yq.sort_index()


def compute_daily_factors(
    df_daily_transformed: pd.DataFrame, 
    n_factors: int = 5, 
    min_non_nan_ratio: float = 0.70,
    exclude_cols: List[str] = None
) -> pd.DataFrame:
    """
    Extract daily factors from transformed financial data using PCA.
    Applies standardization before PCA to ensure all variables contribute equally.
    
    Parameters:
    -----------
    df_daily_transformed : DataFrame
        Transformed daily data (log-returns or first differences)
    n_factors : int
        Number of principal components to extract
    min_non_nan_ratio : float
        Minimum ratio of non-missing values required (default: 70%)
    exclude_cols : List[str]
        Columns to exclude from PCA (e.g., monthly series like NFP)
    
    Notes:
    ------
    NFP (Non-Farm Payrolls) is monthly and should NOT be included in a 
    daily factor extraction. It can be used separately as a macro indicator.
    """
    X = df_daily_transformed.copy()
    
    # Exclude specified columns (e.g., monthly series, target variable)
    exclude_cols = exclude_cols or EXCLUDE_FROM_PCA
    cols_to_drop = [c for c in exclude_cols if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        print(f"  PCA: Excluded {len(cols_to_drop)} series: {cols_to_drop}")

    # Remove columns with too many missing values
    keep_cols = X.notna().mean() >= min_non_nan_ratio
    X = X.loc[:, keep_cols]

    # PCA requires complete data -> drop rows with any NaN
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # Standardize before PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)

    # Extract factors
    pca = PCA(n_components=n_factors, random_state=0)
    F = pca.fit_transform(Xs)

    return pd.DataFrame(F, index=X.index, columns=[f"DF{i+1}" for i in range(n_factors)])
