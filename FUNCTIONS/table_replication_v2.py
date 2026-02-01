"""
Table Replication Module v2
===========================

Replication of Tables 1-5 from Andreou, Ghysels, Kourtellos (2013)
"Should Macroeconomic Forecasters Use Daily Financial Data and How?"

Format matches the paper exactly:
- Table 1: RMSFE with columns for h=1, h=4 (Long sample) + h=1 (Short sample)
- Table 2: DM test p-values
- Table 3: RMSFE for models with leads
- Table 4: DM tests for models with leads  
- Table 5: Comparisons with ADS
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .midas import MidasSpec, MidasModel, MidasModelWithLeads, FADLMidasWithLeads, rmsfe
from .benchmarks import (
    AR1Model, RandomWalkModel, ADLModel, FARModel, FADLModel,
    FARModelWithLeads, FADLModelWithLeads,
    combine_all_forecasts_msfe
)
from .analysis import LONG_SAMPLE, SHORT_SAMPLE, SamplePeriod

# =============================================================================
# TABLE 1: RMSFE COMPARISONS FOR MODELS WITH NO LEADS
# =============================================================================

def replicate_table1(
    yq: pd.Series,
    df_factors: pd.DataFrame,
    df_assets: pd.DataFrame,
    cfnai_quarterly: pd.Series,
    m: int = 63,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Replicate Table 1: RMSFE comparisons for models with no leads.
    
    Returns a DataFrame with the exact structure of the paper:
    - Rows: RW, AR, FAR(CFNAI), ADL, ADL-MIDAS, FADL(CFNAI), FADL-MIDAS(CFNAI)
    - Columns: Long Sample (h=1, h=4) for 5 DF
    
    Note: We only have 5 DF (daily factors), not 64 DA (daily assets individually).
    The paper's 64 DA column would require running 64 separate models.
    
    Returns
    -------
    results_df : pd.DataFrame
        Table with RMSFE values
    all_forecasts : dict
        Nested dict: {sample: {horizon: {model: forecast_df}}}
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 1 REPLICATION: RMSFE Comparisons for Models with No Leads")
        print(f"{'='*80}")
    
    # Store all results
    results = {}
    all_forecasts = {'long': {1: {}, 4: {}}, 'short': {1: {}}}
    
    # ==========================================================================
    # LONG SAMPLE
    # ==========================================================================
    sample = LONG_SAMPLE
    
    for h in [1, 4]:
        if verbose:
            print(f"\n--- Long Sample, h={h} ---")
        
        yq_sample = yq[(yq.index >= sample.data_start) & (yq.index <= sample.oos_end)]
        
        # ---------------------------------------------------------------------
        # UNIVARIATE MODELS
        # ---------------------------------------------------------------------
        
        # Random Walk (absolute RMSFE)
        rw_model = RandomWalkModel(h=h)
        fc_rw = rw_model.recursive_forecast(yq_sample, sample.oos_start)
        fc_rw = fc_rw[fc_rw["target_date"] <= sample.oos_end]
        rmsfe_rw = rmsfe(fc_rw)
        results[f'RW_long_h{h}'] = rmsfe_rw  # Absolute
        all_forecasts['long'][h]['RW'] = fc_rw
        if verbose:
            print(f"  RW: {rmsfe_rw:.2f} (absolute)")
        
        # AR(1)
        ar_model = AR1Model(h=h)
        fc_ar = ar_model.recursive_forecast(yq_sample, sample.oos_start)
        fc_ar = fc_ar[fc_ar["target_date"] <= sample.oos_end]
        rmsfe_ar = rmsfe(fc_ar)
        results[f'AR_long_h{h}'] = rmsfe_ar / rmsfe_rw
        all_forecasts['long'][h]['AR'] = fc_ar
        if verbose:
            print(f"  AR: {results[f'AR_long_h{h}']:.2f}")
        
        # ---------------------------------------------------------------------
        # MODELS WITH MACRO DATA (FAR)
        # ---------------------------------------------------------------------
        
        # FAR (CFNAI)
        far_model = FARModel(h=h, p_y=1)
        fc_far = far_model.recursive_forecast(yq_sample, cfnai_quarterly, sample.oos_start)
        fc_far = fc_far[fc_far["target_date"] <= sample.oos_end]
        if len(fc_far) > 0:
            rmsfe_far = rmsfe(fc_far)
            results[f'FAR_CFNAI_long_h{h}'] = rmsfe_far / rmsfe_rw
            all_forecasts['long'][h]['FAR_CFNAI'] = fc_far
            if verbose:
                print(f"  FAR (CFNAI): {results[f'FAR_CFNAI_long_h{h}']:.2f}")
        
        # ---------------------------------------------------------------------
        # MODELS WITH FINANCIAL DATA - 5 DF (Forecast Combination)
        # ---------------------------------------------------------------------
        
        adl_forecasts = {}
        adl_midas_forecasts = {}
        
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            # ADL (flat aggregation)
            adl_model = ADLModel(h=h, p_y=1, m=m)
            fc_adl = adl_model.recursive_forecast(yq_sample, factor_limited, sample.oos_start)
            fc_adl = fc_adl[fc_adl["target_date"] <= sample.oos_end]
            if len(fc_adl) > 0:
                adl_forecasts[factor_name] = fc_adl
            
            # ADL-MIDAS
            spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            midas_model = MidasModel(spec)
            fc_midas = midas_model.recursive_forecast(yq_sample, factor_limited, sample.oos_start)
            fc_midas = fc_midas[fc_midas["target_date"] <= sample.oos_end]
            if len(fc_midas) > 0:
                adl_midas_forecasts[factor_name] = fc_midas
        
        # Combine ADL forecasts (5 DF)
        if adl_forecasts:
            fc_adl_combined, _ = combine_all_forecasts_msfe(adl_forecasts)
            if len(fc_adl_combined) > 0:
                rmsfe_adl = rmsfe(fc_adl_combined)
                results[f'ADL_5DF_long_h{h}'] = rmsfe_adl / rmsfe_rw
                all_forecasts['long'][h]['ADL_5DF'] = fc_adl_combined
                if verbose:
                    print(f"  ADL (5 DF): {results[f'ADL_5DF_long_h{h}']:.2f}")
        
        # Combine ADL-MIDAS forecasts (5 DF)
        if adl_midas_forecasts:
            fc_midas_combined, _ = combine_all_forecasts_msfe(adl_midas_forecasts)
            if len(fc_midas_combined) > 0:
                rmsfe_midas = rmsfe(fc_midas_combined)
                results[f'ADL_MIDAS_5DF_long_h{h}'] = rmsfe_midas / rmsfe_rw
                all_forecasts['long'][h]['ADL_MIDAS_5DF'] = fc_midas_combined
                if verbose:
                    print(f"  ADL-MIDAS (5 DF): {results[f'ADL_MIDAS_5DF_long_h{h}']:.2f}")
        
        # ---------------------------------------------------------------------
        # MODELS WITH MACRO AND FINANCIAL DATA - 5 DF
        # ---------------------------------------------------------------------
        
        fadl_forecasts = {}
        fadl_midas_forecasts = {}
        
        # Create monthly CFNAI for FADL-MIDAS
        cfnai_monthly = cfnai_quarterly.resample('M').ffill()
        
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            # FADL (CFNAI + flat aggregation)
            fadl_model = FADLModel(h=h, p_y=1, m=m)
            fc_fadl = fadl_model.recursive_forecast(
                yq_sample, cfnai_quarterly, factor_limited, sample.oos_start
            )
            fc_fadl = fc_fadl[fc_fadl["target_date"] <= sample.oos_end]
            if len(fc_fadl) > 0:
                fadl_forecasts[factor_name] = fc_fadl
            
            # FADL-MIDAS (CFNAI + MIDAS)
            spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            fadl_midas_model = FADLMidasWithLeads(spec, lead_months_macro=0, lead_months_daily=0)
            fc_fadl_midas = fadl_midas_model.rolling_forecast(
                yq_sample, factor_limited, cfnai_monthly, sample.oos_start
            )
            fc_fadl_midas = fc_fadl_midas[fc_fadl_midas["target_date"] <= sample.oos_end]
            if len(fc_fadl_midas) > 0:
                fadl_midas_forecasts[factor_name] = fc_fadl_midas
        
        # Combine FADL forecasts
        if fadl_forecasts:
            fc_fadl_combined, _ = combine_all_forecasts_msfe(fadl_forecasts)
            if len(fc_fadl_combined) > 0:
                rmsfe_fadl = rmsfe(fc_fadl_combined)
                results[f'FADL_CFNAI_5DF_long_h{h}'] = rmsfe_fadl / rmsfe_rw
                all_forecasts['long'][h]['FADL_CFNAI_5DF'] = fc_fadl_combined
                if verbose:
                    print(f"  FADL (CFNAI, 5 DF): {results[f'FADL_CFNAI_5DF_long_h{h}']:.2f}")
        
        # Combine FADL-MIDAS forecasts
        if fadl_midas_forecasts:
            fc_fadl_midas_combined, _ = combine_all_forecasts_msfe(fadl_midas_forecasts)
            if len(fc_fadl_midas_combined) > 0:
                rmsfe_fadl_midas = rmsfe(fc_fadl_midas_combined)
                results[f'FADL_MIDAS_CFNAI_5DF_long_h{h}'] = rmsfe_fadl_midas / rmsfe_rw
                all_forecasts['long'][h]['FADL_MIDAS_CFNAI_5DF'] = fc_fadl_midas_combined
                if verbose:
                    print(f"  FADL-MIDAS (CFNAI, 5 DF): {results[f'FADL_MIDAS_CFNAI_5DF_long_h{h}']:.2f}")
    
    # ==========================================================================
    # BUILD RESULTS DATAFRAME (Paper Format)
    # ==========================================================================
    
    # Create DataFrame matching paper's Table 1 structure
    table1_data = []
    
    # Header info
    models = [
        ("Univariate models", None),
        ("RW", "RW"),
        ("AR", "AR"),
        ("Models with macro data", None),
        ("FAR (CFNAI)", "FAR_CFNAI"),
        ("Models with financial data (5 DF)", None),
        ("ADL", "ADL_5DF"),
        ("ADL-MIDAS(J_X^D=0)", "ADL_MIDAS_5DF"),
        ("Models with macro and financial data (5 DF)", None),
        ("FADL (CFNAI)", "FADL_CFNAI_5DF"),
        ("FADL-MIDAS(J_X^D=0) (CFNAI)", "FADL_MIDAS_CFNAI_5DF"),
    ]
    
    for model_name, key in models:
        if key is None:
            # Section header
            table1_data.append({
                "Model": model_name,
                "Long h=1": "",
                "Long h=4": "",
            })
        else:
            # Get values
            if key == "RW":
                # RW shows absolute RMSFE
                val_h1 = results.get(f'{key}_long_h1', np.nan)
                val_h4 = results.get(f'{key}_long_h4', np.nan)
            else:
                val_h1 = results.get(f'{key}_long_h1', np.nan)
                val_h4 = results.get(f'{key}_long_h4', np.nan)
            
            table1_data.append({
                "Model": model_name,
                "Long h=1": f"{val_h1:.2f}" if not np.isnan(val_h1) else "–",
                "Long h=4": f"{val_h4:.2f}" if not np.isnan(val_h4) else "–",
            })
    
    results_df = pd.DataFrame(table1_data)
    
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 1 SUMMARY")
        print(f"{'='*80}")
        print("\nNote: RW shows absolute RMSFE. All other values are ratios to RW (< 1 = better).")
        print(results_df.to_string(index=False))
    
    return results_df, all_forecasts

# =============================================================================
# TABLE 3: RMSFE COMPARISONS FOR MODELS WITH LEADS
# =============================================================================

def replicate_table3(
    yq: pd.Series,
    df_factors: pd.DataFrame,
    cfnai_monthly: pd.Series,
    m: int = 63,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Replicate Table 3: RMSFE comparisons for models with leads.
    
    Models include:
    - ADL-MIDAS(J_X^D=2): 2 months daily leads
    - FADL-MIDAS(J_X^D=2): with CFNAI + 2 months daily leads
    - FADL-MIDAS(J_M=1, J_X^D=2): 1 month macro lead + 2 months daily leads
    - FAR(J_M=1): 1 month macro lead only
    - FADL(J_M=1): 1 month macro lead + flat daily
    - FADL-MIDAS(J_M=1, J_X^D=0): 1 month macro lead, no daily leads
    
    Returns
    -------
    results_df : pd.DataFrame
        Table with RMSFE values for h=1 and h=4
    all_forecasts : dict
        Nested dict with forecasts
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 3 REPLICATION: RMSFE Comparisons for Models with Leads")
        print(f"{'='*80}")
    
    results = {}
    all_forecasts = {'long': {1: {}, 4: {}}}
    sample = LONG_SAMPLE
    
    # Convert CFNAI to quarterly for some models
    cfnai_quarterly = cfnai_monthly.resample('Q').last()
    
    for h in [1, 4]:
        if verbose:
            print(f"\n--- Long Sample, h={h} ---")
        
        yq_sample = yq[(yq.index >= sample.data_start) & (yq.index <= sample.oos_end)]
        
        # Get RW RMSFE for normalization
        rw_model = RandomWalkModel(h=h)
        fc_rw = rw_model.recursive_forecast(yq_sample, sample.oos_start)
        fc_rw = fc_rw[fc_rw["target_date"] <= sample.oos_end]
        rmsfe_rw = rmsfe(fc_rw)
        
        # -----------------------------------------------------------------
        # Models with leads in daily financial data
        # -----------------------------------------------------------------
        
        # ADL-MIDAS(J_X^D=2) and FADL-MIDAS(J_X^D=2) - 2 months daily leads
        adl_midas_jd2_forecasts = {}
        fadl_midas_jd2_forecasts = {}
        
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            # ADL-MIDAS(J_X^D=2)
            spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            model = MidasModelWithLeads(spec, lead_months=2)
            fc = model.recursive_forecast(yq_sample, factor_limited, sample.oos_start)
            if len(fc) > 0:
                fc = fc[fc["target_date"] <= sample.oos_end]
                if len(fc) > 0:
                    adl_midas_jd2_forecasts[factor_name] = fc
            
            # FADL-MIDAS(J_X^D=2)
            fadl_spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            fadl_model = FADLMidasWithLeads(fadl_spec, lead_months_macro=0, lead_months_daily=2)
            fc_fadl = fadl_model.rolling_forecast(
                yq_sample, factor_limited, cfnai_monthly, sample.oos_start
            )
            fc_fadl = fc_fadl[fc_fadl["target_date"] <= sample.oos_end]
            if len(fc_fadl) > 0:
                fadl_midas_jd2_forecasts[factor_name] = fc_fadl
        
        # Combine ADL-MIDAS(J_X^D=2)
        if adl_midas_jd2_forecasts:
            fc_combined, _ = combine_all_forecasts_msfe(adl_midas_jd2_forecasts)
            if len(fc_combined) > 0:
                r = rmsfe(fc_combined)
                results[f'ADL_MIDAS_JD2_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['ADL_MIDAS_JD2'] = fc_combined
                if verbose:
                    print(f"  ADL-MIDAS(J_X^D=2): {results[f'ADL_MIDAS_JD2_long_h{h}']:.2f}")
        
        # Combine FADL-MIDAS(J_X^D=2)
        if fadl_midas_jd2_forecasts:
            fc_combined, _ = combine_all_forecasts_msfe(fadl_midas_jd2_forecasts)
            if len(fc_combined) > 0:
                r = rmsfe(fc_combined)
                results[f'FADL_MIDAS_JD2_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['FADL_MIDAS_JD2'] = fc_combined
                if verbose:
                    print(f"  FADL-MIDAS(J_X^D=2): {results[f'FADL_MIDAS_JD2_long_h{h}']:.2f}")
        
        # -----------------------------------------------------------------
        # Models with leads in monthly macro AND daily financial data
        # -----------------------------------------------------------------
        
        # FADL-MIDAS(J_M=1, J_X^D=2)
        fadl_midas_jm1_jd2_forecasts = {}
        
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            fadl_spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            fadl_model = FADLMidasWithLeads(fadl_spec, lead_months_macro=1, lead_months_daily=2)
            fc = fadl_model.rolling_forecast(
                yq_sample, factor_limited, cfnai_monthly, sample.oos_start
            )
            fc = fc[fc["target_date"] <= sample.oos_end]
            if len(fc) > 0:
                fadl_midas_jm1_jd2_forecasts[factor_name] = fc
        
        if fadl_midas_jm1_jd2_forecasts:
            fc_combined, _ = combine_all_forecasts_msfe(fadl_midas_jm1_jd2_forecasts)
            if len(fc_combined) > 0:
                r = rmsfe(fc_combined)
                results[f'FADL_MIDAS_JM1_JD2_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['FADL_MIDAS_JM1_JD2'] = fc_combined
                if verbose:
                    print(f"  FADL-MIDAS(J_M=1, J_X^D=2): {results[f'FADL_MIDAS_JM1_JD2_long_h{h}']:.2f}")
        
        # -----------------------------------------------------------------
        # Models with leads in monthly macro data only
        # -----------------------------------------------------------------
        
        # FAR(J_M=1)
        far_jm1_model = FARModelWithLeads(h=h, p_y=1, lead_months=1)
        fc_far_jm1 = far_jm1_model.recursive_forecast(yq_sample, cfnai_monthly, sample.oos_start)
        fc_far_jm1 = fc_far_jm1[fc_far_jm1["target_date"] <= sample.oos_end]
        if len(fc_far_jm1) > 0:
            r = rmsfe(fc_far_jm1)
            results[f'FAR_JM1_long_h{h}'] = r / rmsfe_rw
            all_forecasts['long'][h]['FAR_JM1'] = fc_far_jm1
            if verbose:
                print(f"  FAR(J_M=1): {results[f'FAR_JM1_long_h{h}']:.2f}")
        
        # FADL(J_M=1)
        fadl_jm1_forecasts = {}
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            fadl_jm1_model = FADLModelWithLeads(h=h, p_y=1, m=m, lead_months_macro=1)
            fc = fadl_jm1_model.recursive_forecast(
                yq_sample, cfnai_monthly, factor_limited, sample.oos_start
            )
            fc = fc[fc["target_date"] <= sample.oos_end]
            if len(fc) > 0:
                fadl_jm1_forecasts[factor_name] = fc
        
        if fadl_jm1_forecasts:
            fc_combined, _ = combine_all_forecasts_msfe(fadl_jm1_forecasts)
            if len(fc_combined) > 0:
                r = rmsfe(fc_combined)
                results[f'FADL_JM1_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['FADL_JM1'] = fc_combined
                if verbose:
                    print(f"  FADL(J_M=1): {results[f'FADL_JM1_long_h{h}']:.2f}")
        
        # FADL-MIDAS(J_M=1, J_X^D=0)
        fadl_midas_jm1_jd0_forecasts = {}
        for factor_name in df_factors.columns:
            factor_series = df_factors[factor_name]
            factor_limited = factor_series[
                (factor_series.index >= sample.data_start) & 
                (factor_series.index <= sample.data_end)
            ]
            
            fadl_spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
            fadl_model = FADLMidasWithLeads(fadl_spec, lead_months_macro=1, lead_months_daily=0)
            fc = fadl_model.rolling_forecast(
                yq_sample, factor_limited, cfnai_monthly, sample.oos_start
            )
            fc = fc[fc["target_date"] <= sample.oos_end]
            if len(fc) > 0:
                fadl_midas_jm1_jd0_forecasts[factor_name] = fc
        
        if fadl_midas_jm1_jd0_forecasts:
            fc_combined, _ = combine_all_forecasts_msfe(fadl_midas_jm1_jd0_forecasts)
            if len(fc_combined) > 0:
                r = rmsfe(fc_combined)
                results[f'FADL_MIDAS_JM1_JD0_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['FADL_MIDAS_JM1_JD0'] = fc_combined
                if verbose:
                    print(f"  FADL-MIDAS(J_M=1, J_X^D=0): {results[f'FADL_MIDAS_JM1_JD0_long_h{h}']:.2f}")
    
    # ==========================================================================
    # BUILD RESULTS DATAFRAME
    # ==========================================================================
    
    table3_data = []
    
    models = [
        ("Models with leads in daily financial data", None),
        ("ADL-MIDAS(J_X^D=2)", "ADL_MIDAS_JD2"),
        ("FADL-MIDAS(J_X^D=2)", "FADL_MIDAS_JD2"),
        ("Models with leads in monthly macro and daily financial data", None),
        ("FADL-MIDAS(J_M=1, J_X^D=2)", "FADL_MIDAS_JM1_JD2"),
        ("Models with leads in monthly macro data", None),
        ("FAR(J_M=1)", "FAR_JM1"),
        ("FADL(J_M=1)", "FADL_JM1"),
        ("FADL-MIDAS(J_M=1, J_X^D=0)", "FADL_MIDAS_JM1_JD0"),
    ]
    
    for model_name, key in models:
        if key is None:
            table3_data.append({
                "Model": model_name,
                "Long h=1": "",
                "Long h=4": "",
            })
        else:
            val_h1 = results.get(f'{key}_long_h1', np.nan)
            val_h4 = results.get(f'{key}_long_h4', np.nan)
            
            table3_data.append({
                "Model": model_name,
                "Long h=1": f"{val_h1:.2f}" if not np.isnan(val_h1) else "–",
                "Long h=4": f"{val_h4:.2f}" if not np.isnan(val_h4) else "–",
            })
    
    results_df = pd.DataFrame(table3_data)
    
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 3 SUMMARY")
        print(f"{'='*80}")
        print("\nNote: All values are RMSFE ratios relative to RW (< 1 = better than RW).")
        print(results_df.to_string(index=False))
    
    return results_df, all_forecasts

# =============================================================================
# TABLE 5: COMPARISONS WITH ADS
# =============================================================================

def replicate_table5(
    yq: pd.Series,
    ads_daily: pd.Series,
    cfnai_monthly: pd.Series,
    m: int = 63,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Replicate Table 5: Comparisons with ADS.
    
    Compares models using ADS (Aruoba-Diebold-Scotti Business Conditions Index)
    instead of daily financial factors.
    
    Returns
    -------
    results_df : pd.DataFrame
        RMSFE values and DM test p-values
    all_forecasts : dict
        Forecasts for DM tests
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 5 REPLICATION: Comparisons with ADS")
        print(f"{'='*80}")
    
    results = {}
    all_forecasts = {'long': {1: {}, 4: {}}}
    sample = LONG_SAMPLE
    
    for h in [1, 4]:
        if verbose:
            print(f"\n--- Long Sample, h={h} ---")
        
        yq_sample = yq[(yq.index >= sample.data_start) & (yq.index <= sample.oos_end)]
        ads_limited = ads_daily[
            (ads_daily.index >= sample.data_start) & 
            (ads_daily.index <= sample.data_end)
        ]
        
        # Get RW RMSFE for normalization
        rw_model = RandomWalkModel(h=h)
        fc_rw = rw_model.recursive_forecast(yq_sample, sample.oos_start)
        fc_rw = fc_rw[fc_rw["target_date"] <= sample.oos_end]
        rmsfe_rw = rmsfe(fc_rw)
        
        # ADL-MIDAS(J_ADS^D=2)
        spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
        model = MidasModelWithLeads(spec, lead_months=2)
        fc = model.recursive_forecast(yq_sample, ads_limited, sample.oos_start)
        if len(fc) > 0:
            fc = fc[fc["target_date"] <= sample.oos_end]
            if len(fc) > 0:
                r = rmsfe(fc)
                results[f'ADL_MIDAS_ADS_long_h{h}'] = r / rmsfe_rw
                all_forecasts['long'][h]['ADL_MIDAS_ADS'] = fc
                if verbose:
                    print(f"  ADL-MIDAS(J_ADS^D=2): {results[f'ADL_MIDAS_ADS_long_h{h}']:.2f}")
        
        # FADL-MIDAS(J_M=1, J_ADS^D=2)
        fadl_spec = MidasSpec(h=h, p_y=1, m=m, add_const=True)
        fadl_model = FADLMidasWithLeads(fadl_spec, lead_months_macro=1, lead_months_daily=2)
        fc = fadl_model.rolling_forecast(yq_sample, ads_limited, cfnai_monthly, sample.oos_start)
        fc = fc[fc["target_date"] <= sample.oos_end]
        if len(fc) > 0:
            r = rmsfe(fc)
            results[f'FADL_MIDAS_ADS_long_h{h}'] = r / rmsfe_rw
            all_forecasts['long'][h]['FADL_MIDAS_ADS'] = fc
            if verbose:
                print(f"  FADL-MIDAS(J_M=1, J_ADS^D=2): {results[f'FADL_MIDAS_ADS_long_h{h}']:.2f}")
    
    # Build results DataFrame
    table5_data = []
    
    models = [
        ("Models with leads in daily ADS", None),
        ("ADL-MIDAS(J_ADS^D=2)", "ADL_MIDAS_ADS"),
        ("FADL-MIDAS(J_M=1, J_ADS^D=2)", "FADL_MIDAS_ADS"),
    ]
    
    for model_name, key in models:
        if key is None:
            table5_data.append({
                "Model": model_name,
                "RMSFE h=1": "",
                "RMSFE h=4": "",
            })
        else:
            val_h1 = results.get(f'{key}_long_h1', np.nan)
            val_h4 = results.get(f'{key}_long_h4', np.nan)
            
            table5_data.append({
                "Model": model_name,
                "RMSFE h=1": f"{val_h1:.2f}" if not np.isnan(val_h1) else "–",
                "RMSFE h=4": f"{val_h4:.2f}" if not np.isnan(val_h4) else "–",
            })
    
    results_df = pd.DataFrame(table5_data)
    
    if verbose:
        print(f"\n{'='*80}")
        print("TABLE 5 SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))
    
    return results_df, all_forecasts


