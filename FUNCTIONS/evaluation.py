"""
Evaluation Module
=================

Functions for model evaluation, comparison, and sub-period analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .midas import rmsfe


# =============================================================================
# FORECAST COMBINATION
# =============================================================================

def combine_all_forecasts(
    factor_forecasts: Dict[str, pd.DataFrame],
    macro_forecasts: Dict[str, pd.DataFrame],
    combine_func,
    delta: float = 0.9,
    kappa: float = 2.0,
    h: int = 1,
    T0: pd.Timestamp = None,
    verbose: bool = False
) -> Tuple[pd.Series, pd.DataFrame, float, pd.DataFrame, dict]:
    """
    Combine all forecasts (factors + macro) using MSFE weights.
    
    Following Andreou, Ghysels & Kourtellos (2013), Eq. 4.2-4.3.
    
    Parameters
    ----------
    factor_forecasts : Dict[str, pd.DataFrame]
        Forecasts from PCA factors
    macro_forecasts : Dict[str, pd.DataFrame]
        Forecasts from macro indicators
    combine_func : callable
        Function to combine forecasts (e.g., combine_forecasts_msfe)
    delta : float
        Discount factor (default: 0.9 as in paper)
    kappa : float
        Exponent on inverse MSFE (default: 2.0 as in paper's main specification)
    h : int
        Forecast horizon (default: 1 quarter)
    T0 : pd.Timestamp
        Start of OOS period (2001:Q1 for long, 2006:Q1 for short)
    verbose : bool
        If True, print debugging info
        
    Returns
    -------
    tuple: (y_combined, all_forecasts_df, rmsfe_combined, weights_df, diagnostics)
    """
    all_model_forecasts = {**factor_forecasts, **macro_forecasts}
    
    # Find common dates
    common_dates = None
    for name, fc in all_model_forecasts.items():
        dates = set(fc["target_date"])
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    common_dates = sorted(common_dates)
    
    # Build aligned DataFrame
    all_forecasts_df = pd.DataFrame()
    for name, fc in all_model_forecasts.items():
        fc_aligned = fc[fc["target_date"].isin(common_dates)].set_index("target_date")
        all_forecasts_df[name] = fc_aligned["y_pred"]
    
    all_forecasts_df["y_true"] = list(all_model_forecasts.values())[0].set_index("target_date").loc[common_dates, "y_true"]
    
    # Apply MSFE combination with correct parameters
    y_combined, weights_df, diagnostics = combine_func(
        all_forecasts_df, 
        delta=delta, 
        kappa=kappa, 
        h=h,
        T0=T0,
        verbose=verbose
    )
    
    # Calculate combined RMSFE
    valid_idx = ~y_combined.isna()
    y_true_valid = all_forecasts_df.loc[valid_idx, "y_true"]
    y_pred_valid = y_combined[valid_idx]
    rmsfe_combined = np.sqrt(((y_true_valid - y_pred_valid)**2).mean())
    
    return y_combined, all_forecasts_df, rmsfe_combined, weights_df, diagnostics


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def build_summary_table(
    fc_rw: pd.DataFrame,
    fc_ar1: pd.DataFrame,
    factor_forecasts: Dict[str, pd.DataFrame],
    factor_rmsfe: Dict[str, float],
    macro_forecasts: Dict[str, pd.DataFrame],
    macro_rmsfe: Dict[str, float],
    results_by_m: Dict[int, pd.DataFrame],
    rmsfe_rw: float,
    rmsfe_ar1: float,
    rmsfe_combined: float,
    y_true_valid: pd.Series,
    y_pred_valid: pd.Series,
    valid_idx: pd.Series
) -> pd.DataFrame:
    """
    Build summary comparison table for all models.
    """
    summary_data = []
    
    # Random Walk benchmark
    summary_data.append({
        "Model": "Random Walk",
        "Type": "Benchmark",
        "RMSFE": rmsfe_rw,
        "MAE": np.abs(fc_rw["fe"]).mean(),
        "n_forecasts": len(fc_rw),
        "Gain vs RW (%)": 0.0
    })
    
    # AR(1) benchmark
    summary_data.append({
        "Model": "AR(1)",
        "Type": "Benchmark",
        "RMSFE": rmsfe_ar1,
        "MAE": np.abs(fc_ar1["fe"]).mean(),
        "n_forecasts": len(fc_ar1),
        "Gain vs RW (%)": (rmsfe_rw - rmsfe_ar1) / rmsfe_rw * 100
    })
    
    # MSFE Combined MIDAS
    summary_data.append({
        "Model": "MIDAS Combined (MSFE)",
        "Type": "Combined",
        "RMSFE": rmsfe_combined,
        "MAE": np.abs(y_true_valid - y_pred_valid).mean(),
        "n_forecasts": int(valid_idx.sum()),
        "Gain vs RW (%)": (rmsfe_rw - rmsfe_combined) / rmsfe_rw * 100
    })
    
    # MIDAS with Daily Factors (PCA)
    for factor, r in factor_rmsfe.items():
        fc_f = factor_forecasts[factor]
        summary_data.append({
            "Model": f"MIDAS {factor}",
            "Type": "Daily Factor",
            "RMSFE": r,
            "MAE": np.abs(fc_f["fe"]).mean(),
            "n_forecasts": len(fc_f),
            "Gain vs RW (%)": (rmsfe_rw - r) / rmsfe_rw * 100
        })
    
    # MIDAS with Macro Indicators
    for name, r in macro_rmsfe.items():
        fc_m = macro_forecasts[name]
        summary_data.append({
            "Model": f"MIDAS {name}",
            "Type": "Macro Indicator",
            "RMSFE": r,
            "MAE": np.abs(fc_m["fe"]).mean(),
            "n_forecasts": len(fc_m),
            "Gain vs RW (%)": (rmsfe_rw - r) / rmsfe_rw * 100
        })
    
    # MIDAS by window size
    for m, fc_m in results_by_m.items():
        r = rmsfe(fc_m)
        summary_data.append({
            "Model": f"MIDAS m={m} ({m//63}Q)",
            "Type": "Window Size",
            "RMSFE": r,
            "MAE": np.abs(fc_m["fe"]).mean(),
            "n_forecasts": len(fc_m),
            "Gain vs RW (%)": (rmsfe_rw - r) / rmsfe_rw * 100
        })
    
    # Format DataFrame
    df_summary = pd.DataFrame(summary_data).sort_values("RMSFE")
    df_summary["RMSFE"] = df_summary["RMSFE"].round(4)
    df_summary["MAE"] = df_summary["MAE"].round(4)
    df_summary["Gain vs RW (%)"] = df_summary["Gain vs RW (%)"].round(2)
    
    return df_summary


# =============================================================================
# SUB-PERIOD ANALYSIS
# =============================================================================

DEFAULT_SUB_PERIODS = {
    "Full Sample (2000-2025)": (pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31")),
    "Pre-Crisis (2000-2007)": (pd.Timestamp("2000-01-01"), pd.Timestamp("2007-12-31")),
    "Crisis (2008-2009)": (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")),
    "Post-Crisis (2010-2019)": (pd.Timestamp("2010-01-01"), pd.Timestamp("2019-12-31")),
    "COVID & After (2020-2025)": (pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")),
    "Recent (2024-2025)": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")),
}


def compute_rmsfe_period(fc_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, int]:
    """Compute RMSFE for a specific time period."""
    mask = (fc_df["target_date"] >= start) & (fc_df["target_date"] <= end)
    fc_period = fc_df[mask]
    if len(fc_period) < 2:
        return np.nan, 0
    return np.sqrt(fc_period["se"].mean()), len(fc_period)


def analyze_sub_periods(
    models_dict: Dict[str, pd.DataFrame],
    sub_periods: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze forecast performance across sub-periods.
    """
    if sub_periods is None:
        sub_periods = DEFAULT_SUB_PERIODS
    
    results_by_period = []
    
    for period_name, (start, end) in sub_periods.items():
        row = {"Period": period_name}
        
        for model_name, fc_df in models_dict.items():
            if fc_df is None:
                row[model_name] = np.nan
                continue
            rmsfe_val, n_obs = compute_rmsfe_period(fc_df, start, end)
            row[model_name] = rmsfe_val
            if model_name == "Random Walk":
                row["n_obs"] = n_obs
        
        results_by_period.append(row)
    
    df_periods = pd.DataFrame(results_by_period).set_index("Period")
    
    # Round values
    for col in df_periods.columns:
        if col != "n_obs":
            df_periods[col] = df_periods[col].round(4)
    
    # Compute gains vs Random Walk
    df_gains = df_periods.copy()
    for col in df_gains.columns:
        if col not in ["n_obs", "Random Walk"]:
            df_gains[col] = ((df_periods["Random Walk"] - df_periods[col]) / df_periods["Random Walk"] * 100).round(1)
    df_gains = df_gains.drop(columns=["Random Walk", "n_obs"], errors="ignore")
    
    return df_periods, df_gains


# =============================================================================
# RECENT PERIOD ANALYSIS
# =============================================================================

def analyze_recent_forecasts(
    models_dict: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp = pd.Timestamp("2024-01-01")
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze recent quarter-by-quarter forecasts.
    """
    # Get recent forecasts
    recent_forecasts = {}
    for model_name, fc_df in models_dict.items():
        if fc_df is None:
            continue
        mask = fc_df["target_date"] >= start_date
        fc_recent = fc_df[mask].copy()
        if len(fc_recent) > 0:
            recent_forecasts[model_name] = fc_recent
    
    # Get all dates
    all_dates = set()
    for fc in recent_forecasts.values():
        all_dates.update(fc["target_date"].tolist())
    all_dates = sorted(all_dates)
    
    # Build comparison table
    comparison = []
    for date in all_dates:
        q = (date.month - 1) // 3 + 1
        row = {"Quarter": f"{date.year}-Q{q}"}
        
        for model_name, fc_df in recent_forecasts.items():
            fc_date = fc_df[fc_df["target_date"] == date]
            if len(fc_date) > 0:
                if "GDP Actual" not in row:
                    row["GDP Actual"] = fc_date["y_true"].values[0]
                row[model_name] = fc_date["y_pred"].values[0]
        
        comparison.append(row)
    
    df_forecasts = pd.DataFrame(comparison).set_index("Quarter")
    for col in df_forecasts.columns:
        df_forecasts[col] = df_forecasts[col].round(2)
    
    # Compute errors
    df_errors = df_forecasts.copy()
    for col in df_errors.columns:
        if col != "GDP Actual":
            df_errors[col] = (df_forecasts["GDP Actual"] - df_forecasts[col]).round(2)
    df_errors = df_errors.drop(columns=["GDP Actual"])
    
    # Summary statistics
    summary = []
    for model_name, fc in recent_forecasts.items():
        rmsfe_val = np.sqrt(fc["se"].mean())
        mae_val = np.abs(fc["fe"]).mean()
        summary.append({
            "Model": model_name,
            "RMSFE": round(rmsfe_val, 4),
            "MAE": round(mae_val, 4),
            "n": len(fc)
        })
    
    df_summary = pd.DataFrame(summary).sort_values("RMSFE")
    
    return df_forecasts, df_errors, df_summary
