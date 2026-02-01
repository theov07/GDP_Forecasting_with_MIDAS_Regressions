"""
Data Diagnostics Module
========================

Utility functions for data quality checks and diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from .plots import classify_ticker


def compute_coverage_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute coverage statistics by ticker and asset class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with tickers as columns
        
    Returns:
    --------
    coverage_df : pd.DataFrame
        Coverage stats by ticker
    class_summary : pd.DataFrame
        Aggregated stats by asset class
    """
    coverage_info = []
    for col in df.columns:
        non_nan = df[col].count()
        total = len(df)
        coverage = (non_nan / total) * 100
        coverage_info.append({
            'Ticker': col,
            'Coverage': coverage,
            'Class': classify_ticker(col)
        })
    
    coverage_df = pd.DataFrame(coverage_info)
    
    class_summary = coverage_df.groupby('Class').agg({
        'Ticker': 'count',
        'Coverage': 'mean'
    }).round(1)
    class_summary.columns = ['N Tickers', 'Avg Coverage (%)']
    class_summary = class_summary.sort_values('N Tickers', ascending=False)
    
    return coverage_df, class_summary


def diagnose_volatilities(
    df_transformed: pd.DataFrame,
    exclude_cols: set,
    threshold: float = 50.0
) -> Tuple[List[float], List[Tuple[str, float, str]]]:
    """
    Diagnose volatilities and identify extreme cases.
    
    Parameters:
    -----------
    df_transformed : pd.DataFrame
        Transformed data
    exclude_cols : set
        Columns to exclude from analysis
    threshold : float
        Threshold for identifying extreme volatility (default: 50.0%)
        
    Returns:
    --------
    vol_check : List[float]
        All volatility values
    extreme_vol_series : List[Tuple[str, float, str]]
        List of (ticker, volatility, asset_class) for extreme cases
    """
    included_cols = [col for col in df_transformed.columns if col not in exclude_cols]
    
    vol_check = []
    extreme_vol_series = []
    
    for col in included_cols:
        s = df_transformed[col].dropna()
        if len(s) > 100:
            vol = s.std()
            vol_check.append(vol)
            if vol > threshold:
                extreme_vol_series.append((col, vol, classify_ticker(col)))
    
    return vol_check, extreme_vol_series


def print_data_summary(
    df_transformed: pd.DataFrame,
    exclude_cols: set,
    vol_check: List[float],
    extreme_vol_series: List[Tuple[str, float, str]]
) -> None:
    """
    Print formatted data summary with methodology compliance.
    
    Parameters:
    -----------
    df_transformed : pd.DataFrame
        Transformed data
    exclude_cols : set
        Excluded columns
    vol_check : List[float]
        Volatility values
    extreme_vol_series : List[Tuple]
        Extreme volatility series
    """
    included_for_pca = [col for col in df_transformed.columns if col not in exclude_cols]
    
    print("=" * 70)
    print("DATA SUMMARY (Andreou et al. 2013 methodology)")
    print("=" * 70)
    print(f"Total series: {len(df_transformed.columns)}")
    print(f"Included in PCA: {len(included_for_pca)} (all daily financial series)")
    print(f"Excluded: {len(exclude_cols)} ({', '.join(exclude_cols)})")
    
    print(f"\nVolatility distribution after winsorization:")
    print(f"  Median: {np.median(vol_check):.2f}%")
    print(f"  90th percentile: {np.percentile(vol_check, 90):.1f}%")
    print(f"  Max: {np.max(vol_check):.1f}%")
    
    if extreme_vol_series:
        print(f"\n⚠️  EXTREME VOLATILITY SERIES (>50%):")
        for ticker, vol, asset_class in sorted(extreme_vol_series, key=lambda x: x[1], reverse=True):
            print(f"  {ticker:<25} | Vol: {vol:6.1f}% | Class: {asset_class}")
