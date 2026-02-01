"""
MIDAS Forecasting Library
=========================

Implementation of Mixed Data Sampling (MIDAS) regression models for GDP forecasting
using high-frequency financial data, based on Andreou, Ghysels, Kourtellos (2013).

Modules:
--------
- data_loader: Load and parse Bloomberg data
- transformations: Data transformations and PCA factor extraction
- midas: ADL-MIDAS model estimation and recursive forecasting
- benchmarks: AR, ADL, FAR, FADL models and Diebold-Mariano test
- plots: Visualization functions
"""

from .data_loader import load_bloomberg_csv
from .transformations import transform_panel_auto, build_quarterly_target, compute_daily_factors
from .midas import MidasSpec, MidasModel, MidasModelWithLeads, FADLMidasWithLeads, rmsfe
from .benchmarks import (
    AR1Model, 
    RandomWalkModel,
    ADLModel,           # ADL with flat aggregation (Table 1)
    FARModel,           # Factor AR (Table 1)
    FADLModel,          # Factor ADL with flat agg (Table 1)
    FARModelWithLeads,  # FAR with macro leads (Table 3)
    FADLModelWithLeads, # FADL with macro leads (Table 3)
    combine_forecasts_msfe,
    combine_all_forecasts_msfe,  # For 64 DA, 5 DF combinations
    select_ar_lag_aic   # AIC lag selection
)
from .plots import (
    plot_forecast_comparison,
    plot_factor_analysis,
    plot_midas_weights
)


__version__ = "1.0.0"
__author__ = "Gestion Quantitative - M2 272 Dauphine"
