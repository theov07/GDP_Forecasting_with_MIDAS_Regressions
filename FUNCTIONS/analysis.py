"""
Analysis Module
===============

Functions for PCA factor analysis and paper replication following 
Andreou, Ghysels, Kourtellos (2013).

Paper specifications:
- Long sample: Jan 1, 1986 - Dec 31, 2008 (92 quarters, 4584 trading days)
- Short sample: Jan 1, 1999 - Dec 31, 2008 (40 quarters, 1777 trading days)
- Long sample training: 1986:Q1 - 2000:Q4, OOS: 2001:Q1 - 2008:Q4
- Short sample training: 1999:Q1 - 2005:Q4, OOS: 2006:Q1 - 2008:Q4
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .transformations import compute_daily_factors
from .midas import MidasSpec, MidasModel, rmsfe


# =============================================================================
# PAPER SAMPLE DEFINITIONS (Andreou et al. 2013)
# =============================================================================

@dataclass
class SamplePeriod:
    """Sample period definition following the paper."""
    name: str
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    train_end: pd.Timestamp  # End of training period
    oos_start: pd.Timestamp  # Start of out-of-sample
    oos_end: pd.Timestamp    # End of out-of-sample


# Paper sample definitions
LONG_SAMPLE = SamplePeriod(
    name="Long Sample",
    data_start=pd.Timestamp("1986-01-01"),
    data_end=pd.Timestamp("2008-12-31"),
    train_end=pd.Timestamp("2000-12-31"),  # Training: 1986Q1-2000Q4
    oos_start=pd.Timestamp("2001-03-31"),  # OOS starts 2001Q1
    oos_end=pd.Timestamp("2008-12-31"),    # OOS ends 2008Q4
)

SHORT_SAMPLE = SamplePeriod(
    name="Short Sample",
    data_start=pd.Timestamp("1999-01-01"),
    data_end=pd.Timestamp("2008-12-31"),
    train_end=pd.Timestamp("2005-12-31"),  # Training: 1999Q1-2005Q4
    oos_start=pd.Timestamp("2006-03-31"),  # OOS starts 2006Q1
    oos_end=pd.Timestamp("2008-12-31"),    # OOS ends 2008Q4
)


def get_sample_info() -> Dict[str, SamplePeriod]:
    """Return paper sample definitions."""
    return {
        "long": LONG_SAMPLE,
        "short": SHORT_SAMPLE,
    }


# =============================================================================
# PCA FACTOR ANALYSIS (Paper Section 3)
# =============================================================================

def analyze_pca_factors(
    df_transformed: pd.DataFrame,
    df_factors: pd.DataFrame,
    n_factors: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze PCA factors to identify their economic interpretation.
    
    Following Andreou et al. (2013), factors typically represent:
    - Factor 1: Overall market/equity factor
    - Factor 2: Interest rate/fixed income factor
    - Other factors: Sector-specific or volatility factors
    
    Parameters
    ----------
    df_transformed : pd.DataFrame
        Transformed daily financial data
    df_factors : pd.DataFrame
        Extracted PCA factors
    n_factors : int
        Number of factors to analyze
    verbose : bool
        Print analysis results
        
    Returns
    -------
    pd.DataFrame
        Factor loadings and correlations with asset classes
    """
    # Exclude non-daily series
    exclude = ["GDP CQOQ Index", "NFP TCH Index"]
    X = df_transformed.drop(columns=[c for c in exclude if c in df_transformed.columns], errors='ignore')
    
    # Align dates
    common_idx = X.index.intersection(df_factors.index)
    X = X.loc[common_idx]
    factors = df_factors.loc[common_idx]
    
    # Calculate correlations between factors and original series
    correlations = {}
    for col in X.columns:
        if X[col].notna().sum() > 100:
            correlations[col] = {}
            for f in factors.columns:
                corr = X[col].corr(factors[f])
                correlations[col][f] = corr
    
    corr_df = pd.DataFrame(correlations).T
    
    # Classify series by asset class
    asset_classes = {
        "Equity": ["SPX", "NDX", "INDU", "RTY", "VIX", "MXWO", "MXEF"],
        "Fixed Income": ["USGG", "GT", "FDTR", "LIBOR", "OIS"],
        "Credit": ["CDX", "ITRX", "LF98", "LUAC"],
        "Commodities": ["CL", "CO", "GC", "HG", "SI", "NG", "W ", "S ", "C "],
        "FX": ["EUR", "JPY", "GBP", "CHF", "DXY", "BBDXY"],
    }
    
    # Calculate average correlation by asset class
    class_correlations = {}
    for class_name, keywords in asset_classes.items():
        class_series = [col for col in corr_df.index 
                       if any(kw in col.upper() for kw in keywords)]
        if class_series:
            class_correlations[class_name] = corr_df.loc[class_series].mean()
    
    class_corr_df = pd.DataFrame(class_correlations).T
    
    if verbose:
        print("PCA FACTOR ANALYSIS - Economic Interpretation")
        print("=" * 70)
        print("\nAverage correlation by asset class:")
        print(class_corr_df.round(3).to_string())
        
        # Identify dominant factor for each class
        print("\n" + "-" * 70)
        print("Factor interpretation (based on highest correlation):")
        for class_name in class_corr_df.index:
            dominant_factor = class_corr_df.loc[class_name].abs().idxmax()
            corr_value = class_corr_df.loc[class_name, dominant_factor]
            print(f"  {class_name:15s} → {dominant_factor} (r = {corr_value:+.3f})")
        
        print("\n" + "-" * 70)
        print("Following Andreou et al. (2013):")
        print("  - DF1 typically captures overall market/equity movements")
        print("  - DF2 typically captures interest rate/fixed income dynamics")
        print("=" * 70)
    
    return class_corr_df


def compute_pca_loadings(
    df_transformed: pd.DataFrame,
    n_factors: int = 5,
    min_coverage: float = 0.70
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute PCA loadings (eigenvectors) for factor interpretation.
    
    Returns the loadings matrix showing how each original series
    contributes to each factor.
    
    Parameters
    ----------
    df_transformed : pd.DataFrame
        Transformed daily data
    n_factors : int
        Number of factors
    min_coverage : float
        Minimum data coverage required
        
    Returns
    -------
    loadings_df : pd.DataFrame
        Factor loadings for each series
    explained_var : np.ndarray
        Variance explained by each factor
    """
    # Exclude non-daily series
    exclude = ["GDP CQOQ Index", "NFP TCH Index"]
    X = df_transformed.drop(columns=[c for c in exclude if c in df_transformed.columns], errors='ignore')
    
    # Filter by coverage
    keep_cols = X.notna().mean() >= min_coverage
    X = X.loc[:, keep_cols]
    
    # Drop rows with NaN
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # PCA
    pca = PCA(n_components=n_factors, random_state=0)
    pca.fit(X_scaled)
    
    # Loadings = eigenvectors
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"DF{i+1}" for i in range(n_factors)]
    )
    
    return loadings, pca.explained_variance_ratio_


# =============================================================================
# HORIZON ANALYSIS (Paper Section 4)
# =============================================================================

def estimate_multiple_horizons(
    yq: pd.Series,
    df_factors: pd.DataFrame,
    start_oos: pd.Timestamp,
    horizons: List[int] = [1, 2, 3, 4],
    m: int = 63,
    min_train_obs: int = 20,
    verbose: bool = True
) -> Tuple[Dict[int, Dict], pd.DataFrame]:
    """
    Estimate MIDAS models for multiple forecast horizons.
    
    Following Andreou et al. (2013), we test h = 1, 2, 3, 4 quarters.
    
    Parameters
    ----------
    yq : pd.Series
        Quarterly GDP growth
    df_factors : pd.DataFrame
        Daily PCA factors
    start_oos : pd.Timestamp
        Start of out-of-sample period
    horizons : List[int]
        Forecast horizons to test (default: 1-4 quarters)
    m : int
        MIDAS window size in days
    min_train_obs : int
        Minimum training observations
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict[int, Dict]
        Forecasts and RMSFE for each horizon and factor
    summary_df : pd.DataFrame
        Summary table of results
    """
    results = {}
    summary_data = []
    
    if verbose:
        print("MIDAS ESTIMATION - Multiple Horizons")
        print("=" * 70)
    
    for h in horizons:
        spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
        results[h] = {"forecasts": {}, "rmsfe": {}}
        
        if verbose:
            print(f"\nHorizon h = {h} quarter(s):")
        
        for factor in df_factors.columns:
            model = MidasModel(spec)
            fc = model.recursive_forecast(yq, df_factors[factor], start_oos, min_train_obs=min_train_obs)
            
            if fc is not None and len(fc) > 0:
                results[h]["forecasts"][factor] = fc
                results[h]["rmsfe"][factor] = rmsfe(fc)
        
        # Best factor for this horizon
        if results[h]["rmsfe"]:
            best_factor = min(results[h]["rmsfe"], key=results[h]["rmsfe"].get)
            best_rmsfe = results[h]["rmsfe"][best_factor]
            
            if verbose:
                print(f"  Best: {best_factor} (RMSFE = {best_rmsfe:.4f})")
            
            summary_data.append({
                "Horizon": f"h={h}",
                "Best Factor": best_factor,
                "RMSFE": best_rmsfe,
                "N_Forecasts": len(results[h]["forecasts"].get(best_factor, []))
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose:
        print("\n" + "=" * 70)
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    
    return results, summary_df


# =============================================================================
# SAMPLE PERIOD ESTIMATION (Paper replication)
# =============================================================================

def estimate_sample_period(
    yq: pd.Series,
    df_transformed: pd.DataFrame,
    sample: SamplePeriod,
    n_factors: int = 5,
    h: int = 1,
    m: int = 63,
    min_coverage: float = 0.90,
    verbose: bool = True
) -> Dict:
    """
    Estimate MIDAS for a specific sample period following the paper.
    
    This function:
    1. Filters series with sufficient coverage for the sample period
    2. Extracts PCA factors from available data
    3. Estimates MIDAS models with recursive OOS forecasting
    
    Parameters
    ----------
    yq : pd.Series
        Quarterly GDP growth (full sample)
    df_transformed : pd.DataFrame
        Transformed daily data (full sample)
    sample : SamplePeriod
        Sample period definition
    n_factors : int
        Number of PCA factors to extract
    h : int
        Forecast horizon
    m : int
        MIDAS window size
    min_coverage : float
        Minimum coverage required for series inclusion
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict with keys:
        - 'factors': Extracted PCA factors
        - 'forecasts': Dict of forecasts by factor
        - 'rmsfe': Dict of RMSFE by factor
        - 'best_factor': Name of best performing factor
        - 'best_forecast': Forecast DataFrame for best factor
        - 'series_used': List of series used for PCA
    """
    if verbose:
        print(f"\n{sample.name.upper()}")
        print("=" * 70)
        print(f"Data period: {sample.data_start.date()} → {sample.data_end.date()}")
        print(f"Training: until {sample.train_end.date()}")
        print(f"OOS period: {sample.oos_start.date()} → {sample.oos_end.date()}")
    
    # Exclude target and non-daily series
    exclude_cols = ["GDP CQOQ Index", "NFP TCH Index"]
    X = df_transformed.drop(columns=[c for c in exclude_cols if c in df_transformed.columns], errors='ignore')
    
    # Filter series with sufficient coverage since data_start
    series_coverage = {}
    for col in X.columns:
        data_in_period = X.loc[sample.data_start:, col]
        if len(data_in_period) > 0:
            coverage = data_in_period.notna().sum() / len(data_in_period)
            series_coverage[col] = coverage
    
    selected_series = [col for col, cov in series_coverage.items() if cov >= min_coverage]
    
    if verbose:
        print(f"\nSeries with ≥{min_coverage*100:.0f}% coverage: {len(selected_series)}")
    
    if len(selected_series) < n_factors:
        if verbose:
            print(f"ERROR: Only {len(selected_series)} series available (need {n_factors})")
        return None
    
    # Extract PCA factors from selected series
    X_sample = X[selected_series].loc[sample.data_start:]
    df_factors = compute_daily_factors(X_sample, n_factors=n_factors, min_non_nan_ratio=0.70)
    
    if verbose:
        print(f"Factors extracted: {df_factors.shape[0]} days × {df_factors.shape[1]} factors")
        print(f"Factor period: {df_factors.index.min().date()} → {df_factors.index.max().date()}")
    
    # Estimate MIDAS models
    spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
    forecasts = {}
    rmsfe_dict = {}
    
    for factor in df_factors.columns:
        model = MidasModel(spec)
        fc = model.recursive_forecast(yq, df_factors[factor], sample.oos_start, min_train_obs=20)
        
        if fc is not None and len(fc) > 0:
            # Filter to OOS period only
            fc_oos = fc[fc["target_date"] <= sample.oos_end].copy()
            if len(fc_oos) > 0:
                forecasts[factor] = fc_oos
                rmsfe_dict[factor] = rmsfe(fc_oos)
    
    if not rmsfe_dict:
        if verbose:
            print("ERROR: No valid forecasts generated")
        return None
    
    best_factor = min(rmsfe_dict, key=rmsfe_dict.get)
    best_fc = forecasts[best_factor]
    
    if verbose:
        print(f"\nResults:")
        for f, r in sorted(rmsfe_dict.items(), key=lambda x: x[1]):
            marker = "← Best" if f == best_factor else ""
            print(f"  {f}: RMSFE = {r:.4f} {marker}")
        print(f"\nOOS forecasts: {len(best_fc)} ({best_fc['target_date'].min().date()} → {best_fc['target_date'].max().date()})")
        print("=" * 70)
    
    return {
        "factors": df_factors,
        "forecasts": forecasts,
        "rmsfe": rmsfe_dict,
        "best_factor": best_factor,
        "best_forecast": best_fc,
        "series_used": selected_series,
        "sample": sample,
    }


def compare_sample_periods(
    long_results: Dict,
    short_results: Dict,
    fc_ar1: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare results between long and short sample periods.
    
    Returns summary table following the paper's Table format.
    """
    summary = []
    
    for name, results in [("Long Sample", long_results), ("Short Sample", short_results)]:
        if results is None:
            continue
        
        sample = results["sample"]
        best_rmsfe = results["rmsfe"][results["best_factor"]]
        
        # Calculate AR(1) RMSFE for this period
        ar1_oos = fc_ar1[
            (fc_ar1["target_date"] >= sample.oos_start) & 
            (fc_ar1["target_date"] <= sample.oos_end)
        ]
        ar1_rmsfe = rmsfe(ar1_oos) if len(ar1_oos) > 0 else np.nan
        
        gain = (ar1_rmsfe - best_rmsfe) / ar1_rmsfe * 100 if not np.isnan(ar1_rmsfe) else np.nan
        
        summary.append({
            "Sample": name,
            "OOS Period": f"{sample.oos_start.year}-{sample.oos_end.year}",
            "N Series": len(results["series_used"]),
            "N Forecasts": len(results["best_forecast"]),
            "Best Factor": results["best_factor"],
            "MIDAS RMSFE": best_rmsfe,
            "AR(1) RMSFE": ar1_rmsfe,
            "Gain vs AR(1)": f"{gain:+.1f}%",
        })
    
    df = pd.DataFrame(summary)
    
    if verbose:
        print("\nSAMPLE PERIOD COMPARISON")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
    
    return df
