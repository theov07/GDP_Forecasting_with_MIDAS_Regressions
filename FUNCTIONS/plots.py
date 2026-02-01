"""
Visualization Module
====================

Functions for creating analysis plots and saving them to the PLOT ANALYSIS folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Default output directory
PLOT_DIR = Path(__file__).parent.parent / "PLOT ANALYSIS"

# Asset class mapping for grouping tickers
ASSET_CLASSES = {
    "Equity": ["SPX", "NDX", "INDU", "RTY", "CCMP", "VIX"],
    "Fixed Income": ["USGG", "GT", "FDTR", "PRIME", "LIBOR"],
    "Credit Spreads": ["CDX", "ITRX", "LF98", "LUAC"],
    "Commodities": ["CL", "GC", "SI", "HG", "NG", "BCOM"],
    "FX": ["DXY", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD"],
    "Macro": ["GDP", "CFNAI", "ADS", "NAPMPMI", "USURTOT", "CPI", "PCE"],
}


def _ensure_plot_dir():
    """Ensure the plot directory exists."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def classify_ticker(ticker: str) -> str:
    """Classify a ticker into an asset class."""
    ticker_upper = ticker.upper()
    for asset_class, keywords in ASSET_CLASSES.items():
        for kw in keywords:
            if kw in ticker_upper:
                return asset_class
    return "Other"


def plot_data_exploration(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    save: bool = True,
    filename: str = "00_data_exploration.png"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create comprehensive data exploration plots.
    
    Returns:
        coverage_df: Data coverage by ticker
        class_stats: Statistics by asset class
    """
    _ensure_plot_dir()
    
    fig = plt.figure(figsize=(16, 18))
    
    # 1. Data Coverage Timeline
    ax1 = fig.add_subplot(4, 2, 1)
    coverage = df_raw.notna().astype(int)
    # Sample for visualization
    sample_cols = df_raw.columns[:min(30, len(df_raw.columns))]
    coverage_sample = coverage[sample_cols]
    
    # Plot availability as bars
    first_date = []
    last_date = []
    for col in sample_cols:
        valid = df_raw[col].dropna()
        if len(valid) > 0:
            first_date.append(valid.index.min())
            last_date.append(valid.index.max())
        else:
            first_date.append(pd.NaT)
            last_date.append(pd.NaT)
    
    y_pos = range(len(sample_cols))
    for i, (start, end, col) in enumerate(zip(first_date, last_date, sample_cols)):
        if pd.notna(start):
            ax1.barh(i, (end - start).days, left=start, height=0.8, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c[:20] for c in sample_cols], fontsize=7)
    ax1.set_xlabel('Date')
    ax1.set_title('Data Coverage Timeline (first 30 tickers)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Missing Data Heatmap
    ax2 = fig.add_subplot(4, 2, 2)
    # Classify tickers
    classifications = {col: classify_ticker(col) for col in df_raw.columns}
    class_order = ["Equity", "Fixed Income", "Credit Spreads", "Commodities", "FX", "Macro", "Other"]
    
    # Count by class
    class_counts = pd.Series(classifications).value_counts().reindex(class_order).fillna(0)
    colors = plt.cm.Set2(range(len(class_counts)))
    bars = ax2.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
    ax2.set_ylabel('Number of Tickers')
    ax2.set_title('Tickers by Asset Class', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, class_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                 f'{int(val)}', ha='center', fontsize=10, fontweight='bold')
    
    class_stats = []
    for col in df_transformed.columns:
        ac = classify_ticker(col)
        s = df_transformed[col].dropna()
        if len(s) > 10:
            class_stats.append({
                'Ticker': col[:25],
                'Class': ac,
                'Mean': s.mean(),
                'Std': s.std(),
                'Skew': s.skew(),
                'Kurt': s.kurtosis(),
                'N': len(s),
                'Coverage': len(s) / len(df_transformed) * 100
            })
    
    stats_df = pd.DataFrame(class_stats)
    if len(stats_df) > 0:
        # Box plot of standard deviations by class - cap extreme values for visualization
        class_std = stats_df.groupby('Class')['Std'].apply(list).to_dict()
        positions = []
        data = []
        labels = []
        max_std = 20  # More aggressive cap for better visualization (was 50)
        for i, ac in enumerate(class_order):
            if ac in class_std and len(class_std[ac]) > 0:
                positions.append(i)
                # Cap extreme values for better visualization
                capped_stds = [min(std, max_std) for std in class_std[ac]]
                data.append(capped_stds)
                labels.append(ac)
    
    # 4. Correlation Heatmap (sample)
    ax4 = fig.add_subplot(4, 2, 4)
    # Select representative tickers from each class
    sample_tickers = []
    for ac in class_order:
        tickers_in_class = [col for col in df_transformed.columns if classify_ticker(col) == ac]
        if tickers_in_class:
            # Take first 3 from each class
            sample_tickers.extend(tickers_in_class[:3])
    
    if len(sample_tickers) > 3:
        corr_matrix = df_transformed[sample_tickers].corr()
        # Plot correlation heatmap with matplotlib (no seaborn)
        im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(sample_tickers)))
        ax4.set_yticks(range(len(sample_tickers)))
        ax4.set_xticklabels([t[:12] for t in sample_tickers], rotation=45, ha='right', fontsize=7)
        ax4.set_yticklabels([t[:12] for t in sample_tickers], fontsize=7)
        ax4.set_title('Cross-Asset Correlations (sample)', fontweight='bold')
        plt.colorbar(im, ax=ax4, shrink=0.5)
    
    # 5. Time series of returns by asset class (NORMALIZED for comparability)
    ax5 = fig.add_subplot(4, 2, 5)
    # Plot NORMALIZED average return by class - z-score to make scales comparable
    colors_class = {'Equity': 'blue', 'Fixed Income': 'orange', 'Commodities': 'green', 'FX': 'purple'}
    
    for i, ac in enumerate(["Equity", "Fixed Income", "Commodities", "FX"]):
        tickers_in_class = [col for col in df_transformed.columns if classify_ticker(col) == ac]
        
        # Exclude VIX from Equity (it's volatility, not price)
        if ac == "Equity":
            tickers_in_class = [t for t in tickers_in_class if "VIX" not in t.upper()]
        
        if tickers_in_class:
            # For each series, normalize to z-score before averaging
            normalized_series = []
            for ticker in tickers_in_class:
                s = df_transformed[ticker].dropna()
                if len(s) > 50:
                    # Z-score normalization
                    s_norm = (s - s.mean()) / s.std()
                    # Clip extreme values (beyond 3 std) to reduce outlier impact
                    s_norm = s_norm.clip(-3, 3)
                    normalized_series.append(s_norm)
            
            if normalized_series:
                # Combine normalized series - use MEDIAN instead of mean for robustness
                combined_df = pd.concat(normalized_series, axis=1)
                # Use median across series at each timestamp (more robust to outliers)
                avg_return = combined_df.median(axis=1).resample('ME').mean()
                ax5.plot(avg_return.index, avg_return.values, label=ac, 
                        alpha=0.8, lw=1.5, color=colors_class.get(ac, None))
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Normalized Return (z-score, median)')
    ax5.set_title('Asset Class Returns (Normalized, Median across series)', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-1.5, 1.5)  # Tighter limits for median
    
    # 6. Data completeness summary
    ax6 = fig.add_subplot(4, 2, 6)
    # Coverage by year
    df_raw.index = pd.to_datetime(df_raw.index)
    yearly_coverage = df_raw.notna().groupby(df_raw.index.year).mean().mean(axis=1) * 100
    ax6.bar(yearly_coverage.index, yearly_coverage.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.axhline(y=80, color='green', linestyle='--', lw=2, label='80% threshold')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Average Data Coverage (%)')
    ax6.set_title('Data Completeness Over Time', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, 100)
    
    # 7. Key Individual Series (Raw Data)
    ax7 = fig.add_subplot(4, 2, 7)
    key_series_raw = ["SPX Index", "VIX Index", "USGG10YR Index", "DXY Curncy"]
    colors_key = ['blue', 'red', 'green', 'purple']
    plotted_any = False
    for ticker, color in zip(key_series_raw, colors_key):
        # Try to find matching column
        matched_col = None
        for col in df_raw.columns:
            if ticker.upper() in col.upper() or ticker.split()[0].upper() in col.upper():
                matched_col = col
                break
        if matched_col is not None:
            series = df_raw[matched_col].dropna()
            if len(series) > 10:
                # Normalize for comparison (z-score)
                series_norm = (series - series.mean()) / series.std()
                ax7.plot(series_norm.index, series_norm.values, label=matched_col[:20], 
                        alpha=0.8, lw=1.2, color=color)
                plotted_any = True
    if not plotted_any:
        # Fallback: plot first 4 columns with most data
        cols_by_coverage = sorted(df_raw.columns, key=lambda c: df_raw[c].notna().sum(), reverse=True)[:4]
        for i, col in enumerate(cols_by_coverage):
            series = df_raw[col].dropna()
            series_norm = (series - series.mean()) / series.std()
            ax7.plot(series_norm.index, series_norm.values, label=col[:20], alpha=0.8, lw=1.2)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Normalized Value (z-score)')
    ax7.set_title('Key Series - Raw Data (Normalized)', fontweight='bold')
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Key Individual Series (Transformed Data)
    ax8 = fig.add_subplot(4, 2, 8)
    key_series_trans = ["SPX", "VIX", "USGG", "DXY"]
    colors_key = ['blue', 'red', 'green', 'purple']
    plotted_any = False
    for ticker, color in zip(key_series_trans, colors_key):
        matched_col = None
        for col in df_transformed.columns:
            if ticker.upper() in col.upper():
                matched_col = col
                break
        if matched_col is not None:
            series = df_transformed[matched_col].dropna()
            if len(series) > 10:
                # Resample to monthly for cleaner viz
                series_monthly = series.resample('ME').mean()
                ax8.plot(series_monthly.index, series_monthly.values, label=matched_col[:20], 
                        alpha=0.8, lw=1.5, color=color)
                plotted_any = True
    if not plotted_any:
        # Fallback: plot first 4 columns with most data
        cols_by_coverage = sorted(df_transformed.columns, key=lambda c: df_transformed[c].notna().sum(), reverse=True)[:4]
        for i, col in enumerate(cols_by_coverage):
            series = df_transformed[col].dropna().resample('ME').mean()
            ax8.plot(series.index, series.values, label=col[:20], alpha=0.8, lw=1.5)
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Transformed Value (%)')
    ax8.set_title('Key Series - Transformed Data (Monthly Avg)', fontweight='bold')
    ax8.legend(loc='upper left', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {PLOT_DIR / filename}")
    
    plt.show()
    
    # Create summary DataFrames
    coverage_df = pd.DataFrame({
        'Ticker': df_raw.columns,
        'Class': [classify_ticker(c) for c in df_raw.columns],
        'First': [df_raw[c].dropna().index.min() if df_raw[c].notna().any() else pd.NaT for c in df_raw.columns],
        'Last': [df_raw[c].dropna().index.max() if df_raw[c].notna().any() else pd.NaT for c in df_raw.columns],
        'Coverage': [df_raw[c].notna().sum() / len(df_raw) * 100 for c in df_raw.columns]
    }).sort_values(['Class', 'Ticker'])
    
    return coverage_df, stats_df


def plot_forecast_comparison(
    fc_midas: pd.DataFrame,
    fc_ar1: pd.DataFrame,
    results_by_m: Dict[int, pd.DataFrame],
    rmsfe_ar1: float,
    save: bool = True,
    filename: str = "01_forecast_comparison.png"
) -> None:
    """
    Create comparison plots: MIDAS vs AR(1), errors, and m sensitivity.
    """
    _ensure_plot_dir()
    
    def rmsfe(df):
        return float(np.sqrt(df["se"].mean())) if len(df) > 0 else np.nan
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Forecasts vs Actual
    ax1 = axes[0, 0]
    ax1.plot(fc_midas["target_date"], fc_midas["y_true"], 'k-', lw=2, label='GDP Actual', alpha=0.8)
    ax1.plot(fc_midas["target_date"], fc_midas["y_pred"], 'b--', lw=1.5, 
             label=f'MIDAS (RMSFE={rmsfe(fc_midas):.2f})')
    ax1.plot(fc_ar1["target_date"], fc_ar1["y_pred"], 'r:', lw=1.5, 
             label=f'AR(1) (RMSFE={rmsfe_ar1:.2f})')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_title('GDP Forecasts: MIDAS vs AR(1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('GDP Growth (%)')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast Errors
    ax2 = axes[0, 1]
    ax2.bar(fc_midas["target_date"], fc_midas["fe"], alpha=0.6, label='MIDAS', color='blue', width=60)
    ax2.bar(fc_ar1["target_date"], fc_ar1["fe"], alpha=0.4, label='AR(1)', color='red', width=60)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Forecast Errors', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSFE by m
    ax3 = axes[1, 0]
    m_values = sorted(results_by_m.keys())
    rmsfe_values = [rmsfe(results_by_m[m]) for m in m_values]
    colors = ['green' if r == min(rmsfe_values) else 'steelblue' for r in rmsfe_values]
    bars = ax3.bar(range(len(m_values)), rmsfe_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(m_values)))
    ax3.set_xticklabels([f"{m//63}Q\n({m}d)" for m in m_values])
    ax3.axhline(y=rmsfe_ar1, color='red', linestyle='--', lw=2, label=f'AR(1) = {rmsfe_ar1:.2f}')
    ax3.set_xlabel('MIDAS Window (quarters of daily data)')
    ax3.set_ylabel('RMSFE')
    ax3.set_title('Impact of Window Size m on Performance', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmsfe_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Forecasts with different m
    ax4 = axes[1, 1]
    ax4.plot(fc_midas["target_date"], fc_midas["y_true"], 'k-', lw=2.5, label='GDP Actual')
    linestyles = ['-', '--', '-.', ':']
    colors_m = ['blue', 'green', 'orange', 'purple']
    for i, m_val in enumerate(m_values[:4]):
        fc_m = results_by_m[m_val]
        ax4.plot(fc_m["target_date"], fc_m["y_pred"], linestyle=linestyles[i], 
                 color=colors_m[i], lw=1.5, alpha=0.8,
                 label=f'm={m_val} ({m_val//63}Q)')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_title('MIDAS Forecasts with Different Window Sizes', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('GDP Growth (%)')
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR / filename}")
    
    plt.show()


def plot_factor_analysis(
    fc_midas: pd.DataFrame,
    fc_ar1: pd.DataFrame,
    factor_rmsfe: Dict[str, float],
    rmsfe_ar1: float,
    save: bool = True,
    filename: str = "02_factor_analysis.png"
) -> None:
    """
    Create factor analysis plots: performance by factor, theta evolution, crisis periods.
    """
    _ensure_plot_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RMSFE by factor
    ax1 = axes[0, 0]
    factors = list(factor_rmsfe.keys())
    rmsfe_vals = list(factor_rmsfe.values())
    colors = ['green' if r == min(rmsfe_vals) else 'steelblue' for r in rmsfe_vals]
    bars = ax1.bar(factors, rmsfe_vals, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=rmsfe_ar1, color='red', linestyle='--', lw=2, label=f'AR(1) = {rmsfe_ar1:.2f}')
    ax1.set_xlabel('PCA Factor')
    ax1.set_ylabel('RMSFE')
    ax1.set_title('Performance by PCA Factor', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmsfe_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', 
                 ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Theta evolution
    ax2 = axes[0, 1]
    ax2.plot(fc_midas["target_date"], fc_midas["theta"], 'b-', lw=1.5, marker='o', markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=fc_midas["theta"].mean(), color='red', linestyle='-', alpha=0.7,
                label=f'Mean = {fc_midas["theta"].mean():.3f}')
    ax2.set_title('θ Parameter Evolution (MIDAS Weight Decay)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('θ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 2008 Crisis zoom
    ax3 = axes[1, 0]
    mask = (fc_midas["target_date"] >= "2007-01-01") & (fc_midas["target_date"] <= "2012-01-01")
    fc_crisis = fc_midas[mask]
    
    if len(fc_crisis) > 0:
        # Get the date range of MIDAS data
        midas_min_date = fc_crisis["target_date"].min()
        midas_max_date = fc_crisis["target_date"].max()
        
        # Restrict AR(1) to the same period as MIDAS
        fc_ar1_crisis = fc_ar1[(fc_ar1["target_date"] >= midas_min_date) & 
                               (fc_ar1["target_date"] <= midas_max_date)]
        
        ax3.plot(fc_crisis["target_date"], fc_crisis["y_true"], 'k-', lw=2, label='GDP Actual', marker='s', markersize=4)
        ax3.plot(fc_crisis["target_date"], fc_crisis["y_pred"], 'b--', lw=1.5, label='MIDAS', marker='o', markersize=4)
        ax3.plot(fc_ar1_crisis["target_date"], fc_ar1_crisis["y_pred"], 'r:', lw=1.5, label='AR(1)', marker='^', markersize=4)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-06-01"), alpha=0.2, color='red', label='Recession')
        ax3.legend(loc='lower left', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No MIDAS forecasts\nin this period', 
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    ax3.set_title('Zoom: 2008-2009 Financial Crisis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('GDP Growth (%)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: COVID zoom
    ax4 = axes[1, 1]
    mask = (fc_midas["target_date"] >= "2019-01-01") & (fc_midas["target_date"] <= "2022-01-01")
    fc_covid = fc_midas[mask]
    
    if len(fc_covid) > 0:
        # Get the date range of MIDAS data
        midas_min_date = fc_covid["target_date"].min()
        midas_max_date = fc_covid["target_date"].max()
        
        # Restrict AR(1) to the same period
        fc_ar1_covid = fc_ar1[(fc_ar1["target_date"] >= midas_min_date) & 
                              (fc_ar1["target_date"] <= midas_max_date)]
        
        ax4.plot(fc_covid["target_date"], fc_covid["y_true"], 'k-', lw=2, label='GDP Actual', marker='s', markersize=4)
        ax4.plot(fc_covid["target_date"], fc_covid["y_pred"], 'b--', lw=1.5, label='MIDAS', marker='o', markersize=4)
        if len(fc_ar1_covid) > 0:
            ax4.plot(fc_ar1_covid["target_date"], fc_ar1_covid["y_pred"], 'r:', lw=1.5, label='AR(1)', marker='^', markersize=4)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax4.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01"), alpha=0.2, color='red', label='COVID')
        ax4.legend(loc='lower left', fontsize=9)
    else:
        # Display informative message when no data
        ax4.text(0.5, 0.5, 'No MIDAS forecasts after 2008\n\n(Paper methodology:\nOOS period ends 2008:Q4)', 
                 transform=ax4.transAxes, ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('Zoom: COVID-19 Crisis 2020', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('GDP Growth (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR / filename}")
    
    plt.show()


def plot_midas_weights(
    fc_midas: pd.DataFrame,
    m: int = 63,
    save: bool = True,
    filename: str = "03_midas_weights.png"
) -> None:
    """
    Visualize MIDAS weighting scheme and theta distribution.
    """
    _ensure_plot_dir()
    
    def exp_almon_weights(m, theta):
        """
        Weights with correct indexing:
        - Position 0 = oldest, position m-1 = most recent
        - theta < 0 → more weight on RECENT (higher position index)
        """
        # k goes from m-1 (for oldest) to 0 (for most recent)
        # So w[m-1] = exp(theta * 0) = 1 (max when theta < 0)
        k = np.arange(m-1, -1, -1)
        theta = np.clip(theta, -1.0, 1.0)
        a = np.exp(theta * k)
        return a / a.sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Weight functions for different theta
    ax1 = axes[0]
    thetas = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05]
    for theta in thetas:
        w = exp_almon_weights(m, theta)
        ax1.plot(range(m), w, lw=1.5, label=f'θ = {theta}')
    ax1.set_xlabel('Day index (0 = most recent, 62 = oldest)')
    ax1.set_ylabel('Normalized Weight')
    ax1.set_title('Exponential Almon Weights (θ < 0 → more weight on recent)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of estimated theta
    ax2 = axes[1]
    thetas = fc_midas["theta"].values
    ax2.hist(thetas, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=thetas.mean(), color='red', linestyle='--', lw=2, label=f'Mean = {thetas.mean():.3f}')
    ax2.axvline(x=np.median(thetas), color='orange', linestyle=':', lw=2, label=f'Median = {np.median(thetas):.3f}')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Estimated θ')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Estimated θ (Rolling Estimation)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR / filename}")
    
    plt.show()
    
    # Print theta statistics
    print(f"\nθ Statistics (m={m}):")
    print(f"  Mean:     {thetas.mean():.4f}")
    print(f"  Median:   {np.median(thetas):.4f}")
    print(f"  Std Dev:  {thetas.std():.4f}")
    print(f"  % < 0:    {100 * (thetas < 0).mean():.1f}%")
    
    if thetas.mean() < 0:
        print("\n→ θ < 0: Recent observations receive more weight (exponential decay)")


def plot_state_dependent_analysis(
    fc_state_dep: pd.DataFrame,
    fc_standard: pd.DataFrame,
    m: int = 63,
    save: bool = True,
    filename: str = "04_state_dependent_midas.png"
) -> None:
    """
    Visualize state-dependent MIDAS extension results.
    
    Creates a 4-panel figure:
    1. Time-varying theta_t
    2. VIX vs theta_t scatter
    3. Forecast comparison (standard vs state-dependent)
    4. Weight profiles by stress level
    """
    _ensure_plot_dir()
    
    from .midas import exp_almon_weights
    
    if len(fc_state_dep) == 0:
        print("No state-dependent forecasts to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time-varying theta
    ax1 = axes[0, 0]
    ax1.plot(fc_state_dep["forecast_origin"], fc_state_dep["theta_t"], 
             'b-', linewidth=1.5, label="θ_t (effective)")
    ax1.axhline(y=fc_state_dep["theta0"].mean(), color='r', linestyle='--', 
                label=f"θ₀ (baseline) = {fc_state_dep['theta0'].mean():.3f}")
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel("Forecast Origin")
    ax1.set_ylabel("θ_t")
    ax1.set_title("Time-Varying Weight Decay Parameter", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. VIX vs theta_t scatter
    ax2 = axes[0, 1]
    ax2.scatter(fc_state_dep["z_t"], fc_state_dep["theta_t"], alpha=0.6, s=30)
    # Regression line
    z_fit = np.polyfit(fc_state_dep["z_t"], fc_state_dep["theta_t"], 1)
    p = np.poly1d(z_fit)
    x_line = np.linspace(fc_state_dep["z_t"].min(), fc_state_dep["z_t"].max(), 100)
    ax2.plot(x_line, p(x_line), "r--", linewidth=2, label=f"slope = {z_fit[0]:.4f}")
    ax2.set_xlabel("Standardized VIX (Z_t)")
    ax2.set_ylabel("θ_t")
    ax2.set_title("Stress Indicator vs Weight Decay", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Forecast comparison
    ax3 = axes[1, 0]
    ax3.plot(fc_standard["target_date"], fc_standard["y_true"], 'k-', 
             linewidth=2, label="Actual GDP")
    ax3.plot(fc_standard["target_date"], fc_standard["y_pred"], 'b--', 
             alpha=0.8, label="Standard MIDAS")
    ax3.plot(fc_state_dep["target_date"], fc_state_dep["y_pred"], 'r--', 
             alpha=0.8, label="State-Dep MIDAS")
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("GDP Growth (%)")
    ax3.set_title("Forecast Comparison", fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Weight profiles by stress level
    ax4 = axes[1, 1]
    k = np.arange(1, m + 1)
    
    theta0_mean = fc_state_dep["theta0"].mean()
    theta1_mean = fc_state_dep["theta1"].mean()
    
    for z_level, label, color in [(-1, "Low stress (Z=-1)", "green"), 
                                   (0, "Normal (Z=0)", "blue"),
                                   (2, "High stress (Z=2)", "red")]:
        theta_eff = theta0_mean + theta1_mean * z_level
        w = exp_almon_weights(m, theta_eff)
        ax4.plot(k, w, label=f"{label}: θ={theta_eff:.3f}", color=color, linewidth=2)
    
    ax4.set_xlabel("Lag (days)")
    ax4.set_ylabel("Weight")
    ax4.set_title("Weight Profiles by Stress Level", fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR / filename}")
    
    plt.show()
    
    # Print summary
    print(f"\nState-Dependent MIDAS Parameter Summary:")
    print(f"  θ₀ (baseline):  {theta0_mean:.4f}")
    print(f"  θ₁ (stress):    {theta1_mean:.4f}")
    print(f"  Correlation(VIX, θ_t): {z_fit[0]:.4f}")


def plot_period_forecast(
    fc_midas: pd.DataFrame,
    fc_ar1: pd.DataFrame,
    start_date: str,
    end_date: str,
    period_label: str,
    save: bool = True,
    filename: Optional[str] = None
) -> None:
    """
    Plot GDP forecasts for a specific time period (replicating paper's analysis).
    
    Parameters:
    -----------
    fc_midas : pd.DataFrame
        MIDAS forecast results
    fc_ar1 : pd.DataFrame
        AR(1) forecast results
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    period_label : str
        Label for the period (e.g., "Long Period", "Short Period")
    save : bool
        Whether to save the plot
    filename : str, optional
        Output filename (auto-generated if not provided)
    """
    _ensure_plot_dir()
    
    def rmsfe(df):
        return float(np.sqrt(df["se"].mean())) if len(df) > 0 else np.nan
    
    # Filter data for the period
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    fc_midas_period = fc_midas[(fc_midas["target_date"] >= start) & (fc_midas["target_date"] <= end)]
    fc_ar1_period = fc_ar1[(fc_ar1["target_date"] >= start) & (fc_ar1["target_date"] <= end)]
    
    # Compute period-specific RMSFE
    rmsfe_midas = rmsfe(fc_midas_period)
    rmsfe_ar1 = rmsfe(fc_ar1_period)
    improvement = ((rmsfe_ar1 - rmsfe_midas) / rmsfe_ar1) * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual and forecasts
    ax.plot(fc_midas_period["target_date"], fc_midas_period["y_true"], 
            'k-', lw=2.5, label='GDP Actual', marker='o', markersize=5, alpha=0.9)
    ax.plot(fc_midas_period["target_date"], fc_midas_period["y_pred"], 
            'b--', lw=2, label=f'MIDAS (RMSFE={rmsfe_midas:.2f})', 
            marker='s', markersize=4, alpha=0.8)
    ax.plot(fc_ar1_period["target_date"], fc_ar1_period["y_pred"], 
            'r:', lw=2, label=f'AR(1) (RMSFE={rmsfe_ar1:.2f})', 
            marker='^', markersize=4, alpha=0.8)
    
    # Add zero line and recession shading
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Shade recession periods (2008-2009, 2020 COVID)
    if start <= pd.Timestamp("2009-12-31") and end >= pd.Timestamp("2007-12-01"):
        ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"), 
                   alpha=0.1, color='red', label='Financial Crisis')
    if start <= pd.Timestamp("2020-12-31") and end >= pd.Timestamp("2020-03-01"):
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), 
                   alpha=0.1, color='orange', label='COVID-19')
    
    # Labels and formatting
    ax.set_title(f'GDP Growth Forecasts: {period_label}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('GDP Growth (%)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add performance text box
    textstr = f'MIDAS Improvement: {improvement:+.1f}%\nObservations: {len(fc_midas_period)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    if filename is None:
        filename = f"06_forecast_{period_label.lower().replace(' ', '_')}.png"
    
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR / filename}")
    
    plt.show()
    
    # Print summary
    print(f"\n{period_label} Forecast Performance ({start.date()} to {end.date()}):")
    print(f"  MIDAS RMSFE:      {rmsfe_midas:.4f}")
    print(f"  AR(1) RMSFE:      {rmsfe_ar1:.4f}")
    print(f"  Improvement:      {improvement:+.1f}%")
    print(f"  Observations:     {len(fc_midas_period)}")
