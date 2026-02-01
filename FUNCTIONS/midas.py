"""
MIDAS Model Module
==================

Implementation of the ADL-MIDAS (Autoregressive Distributed Lag - Mixed Data Sampling)
model following Andreou, Ghysels, Kourtellos (2013).

The model specification is:
    y_{t+h} = α + ρ·y_t + β·Σ_{k=0}^{m-1} B(k;θ)·x_{t-k} + ε

Where:
- h: forecast horizon (in quarters)
- p_y: number of AR lags
- m: number of daily observations in the MIDAS block
- B(k;θ): exponential Almon weighting function
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares


@dataclass
class MidasSpec:
    """
    Specification for ADL-MIDAS model.
    """
    h: int = 1
    p_y: int = 1
    m: int = 63
    add_const: bool = True


def exp_almon_weights(m: int, theta: float) -> np.ndarray:
    """
    Compute normalized exponential Almon weights.
    B(k; θ) = exp(θ·k) / Σ exp(θ·j)
    
    Paper convention (Andreou et al. 2013):
    - θ < 0 → more weight on RECENT observations
    - θ > 0 → more weight on OLDER observations
    - θ = 0 → uniform weights
    
    Index mapping:
    - k = 0 corresponds to MOST RECENT observation (block[-1])
    - k = m-1 corresponds to OLDEST observation (block[0])
    
    Returns array aligned with block: w[0] for block[0] (oldest), w[-1] for block[-1] (recent)
    """
    k = np.arange(m - 1, -1, -1)  # k = m-1 (oldest) down to 0 (most recent)
    theta = np.clip(theta, -1.0, 1.0)  # Avoid numerical overflow
    a = np.exp(theta * k)
    return a / a.sum()


def midas_aggregate(block: np.ndarray, theta: float) -> float:
    """
    Aggregate a block of daily data using MIDAS weights.
    """
    m = len(block)
    w = exp_almon_weights(m, theta)
    
    # Handle NaN values by renormalizing weights
    valid = ~np.isnan(block)
    if valid.sum() == 0:
        return 0.0
    
    w_valid = w[valid]
    w_valid = w_valid / w_valid.sum()
    return float(np.dot(w_valid, block[valid]))


def rmsfe(df_fc: pd.DataFrame) -> float:
    """
    Compute Root Mean Squared Forecast Error.
    """
    if len(df_fc) == 0 or df_fc["se"].isna().all():
        return np.nan
    return float(np.sqrt(df_fc["se"].mean()))


class MidasModel:
    """
    ADL-MIDAS Model for GDP Forecasting.
    """
    
    def __init__(self, spec: MidasSpec):
        """Initialize MIDAS model with given specification."""
        self.spec = spec
        self.params = None
        self.fitted = False
    
    def _extract_last_m(self, series: pd.Series, end_date: pd.Timestamp) -> np.ndarray:
        """Extract the last m observations up to end_date."""
        s = series.loc[:end_date].dropna()
        arr = np.full(self.spec.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.spec.m:].values
        arr[-len(take):] = take
        return arr
    
    def _build_rows(self, yq: pd.Series, x_daily: pd.Series) -> Tuple[List, np.ndarray, List]:
        """
        Build training rows for MIDAS estimation.
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            # Target date: t + h quarters
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            # Check AR lags availability
            if i < self.spec.p_y:
                continue
            
            # Collect AR lags
            y_lags = []
            for j in range(self.spec.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.spec.p_y:
                continue
            
            # Extract daily block
            block = self._extract_last_m(x_daily, t)
            
            # Require at least 50% valid data
            valid_count = (~np.isnan(block)).sum()
            if valid_count < self.spec.m * 0.5:
                continue
            
            rows.append((y_lags, block))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, x_daily: pd.Series, theta_init: float = -0.05) -> bool:
        """
        Fit the MIDAS model using nonlinear least squares.
        """
        rows, y, _ = self._build_rows(yq, x_daily)
        
        if len(rows) < 10:
            return False
        
        n_params = (1 if self.spec.add_const else 0) + self.spec.p_y + 2  # const + rho + beta + theta
        
        def unpack(b):
            pos = 0
            c = b[pos] if self.spec.add_const else 0.0
            pos += (1 if self.spec.add_const else 0)
            rho = b[pos:pos+self.spec.p_y]
            pos += self.spec.p_y
            beta = b[pos]
            theta = b[pos+1]
            return c, rho, beta, theta
        
        def residuals(b):
            c, rho, beta, theta = unpack(b)
            yhat = np.zeros(len(y))
            for i, (y_lags, block) in enumerate(rows):
                pred = c + float(np.dot(rho, np.array(y_lags)))
                pred += beta * midas_aggregate(block, theta)
                yhat[i] = pred
            return y - yhat
        
        # Initialize parameters
        b0 = np.zeros(n_params)
        pos = 0
        if self.spec.add_const:
            b0[pos] = np.mean(y)
            pos += 1
        b0[pos:pos+self.spec.p_y] = 0.3
        pos += self.spec.p_y
        b0[pos] = 0.0  # beta
        b0[pos+1] = theta_init
        
        # Bounds
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)
        lb[-1], ub[-1] = -1.0, 1.0  # theta bounds
        
        try:
            res = least_squares(residuals, b0, bounds=(lb, ub), method="trf", max_nfev=1000)
            c, rho, beta, theta = unpack(res.x)
            self.params = {
                "const": float(c),
                "rho": rho.astype(float),
                "beta": float(beta),
                "theta": float(theta)
            }
            self.fitted = res.success
            return self.fitted
        except Exception:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts.
        
        At each quarter t >= start_date:
        1. Estimate model on [beginning, t]
        2. Forecast y_{t+h}
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Fit on training data
            y_train = yq.loc[:train_end]
            rows, y_vec, _ = self._build_rows(y_train, x_daily)
            
            if len(rows) < min_train_obs:
                continue
            
            # Estimate
            model = MidasModel(self.spec)
            if not model.fit(y_train, x_daily):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.spec.p_y) if i-j >= 0]
            if len(y_lags) != self.spec.p_y:
                continue
            
            block = self._extract_last_m(x_daily, train_end)
            if (~np.isnan(block)).sum() < self.spec.m * 0.5:
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["beta"] * midas_aggregate(block, model.params["theta"])
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "theta": model.params["theta"],
                "beta": model.params["beta"],
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# MIDAS MODEL WITH LEADS (Andreou et al. 2013, Equation 2.5)
# =============================================================================

class MidasModelWithLeads:
    """
    ADL-MIDAS Model WITH LEADS following Andreou et al. (2013).
    
    Key difference from standard MIDAS:
    - When forecasting GDP for quarter t+1, we use:
      - LAGS: daily data from quarters t, t-1, ... (standard)
      - LEADS: daily data observed within quarter t+1 up to info_date
    
    The paper's "information date" is typically the end of month 2 of quarter t+1,
    giving approximately 44 trading days of leads.
    
    Parameters:
    -----------
    lead_months : int
        Number of months of leads to include (1 or 2, paper uses 2)
    lead_days : int
        Alternative: exact number of trading days as leads (default ~44 for 2 months)
    """
    
    def __init__(self, spec: MidasSpec, lead_months: int = 2):
        """
        Initialize MIDAS model with leads.
        
        Args:
            spec: MidasSpec with h, p_y, m parameters
            lead_months: Number of months of leads (1 or 2). Paper uses 2.
        """
        self.spec = spec
        self.lead_months = lead_months
        # Approximate trading days per month = 21
        self.lead_days = lead_months * 21  # ~42-44 days for 2 months
        self.params = None
        self.fitted = False
    
    def _compute_info_date(self, target_quarter_end: pd.Timestamp) -> pd.Timestamp:
        """
        Compute the information date for forecasting a given quarter.
        
        Info date = end of month (lead_months) within the target quarter.
        
        Example for target Q2 2001 (end = 2001-06-30):
        - lead_months=2 → info_date = 2001-05-31 (end of May)
        - lead_months=1 → info_date = 2001-04-30 (end of April)
        """
        # Get the first day of the target quarter
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        
        # Info date = end of month (lead_months) within the quarter
        info_month = quarter_start + pd.DateOffset(months=self.lead_months - 1)
        # Go to last day of that month
        info_date = info_month + pd.offsets.MonthEnd(0)
        
        return info_date
    
    def _extract_block_with_leads(
        self, 
        x_daily: pd.Series, 
        last_known_quarter_end: pd.Timestamp,
        info_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Extract daily block including leads.
        
        Structure of the block:
        [LAGS from past quarters | LEADS from current quarter]
        
        Args:
            x_daily: Daily factor series
            last_known_quarter_end: End of the last quarter with known GDP
            info_date: Information date (end of month 2 in target quarter)
        
        Returns:
            Array of length (m + lead_days) with lags + leads
        """
        # Total block size: lags + leads
        total_m = self.spec.m + self.lead_days
        
        # Extract all data up to info_date
        s = x_daily.loc[:info_date].dropna()
        
        arr = np.full(total_m, np.nan)
        if len(s) == 0:
            return arr
        
        take = s.iloc[-total_m:].values
        arr[-len(take):] = take
        
        return arr
    
    def _exp_almon_weights_extended(self, total_m: int, theta: float) -> np.ndarray:
        """
        Compute exponential Almon weights for extended block (lags + leads).
        
        The paper uses a SINGLE theta for both lags and leads.
        Index k goes from 0 (oldest lag) to total_m-1 (most recent lead).
        """
        k = np.arange(total_m)
        theta = np.clip(theta, -1.0, 1.0)
        a = np.exp(theta * k)
        return a / a.sum()
    
    def _midas_aggregate_extended(self, block: np.ndarray, theta: float) -> float:
        """Aggregate extended block (lags + leads) using MIDAS weights."""
        total_m = len(block)
        w = self._exp_almon_weights_extended(total_m, theta)
        
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return 0.0
        
        w_valid = w[valid]
        w_valid = w_valid / w_valid.sum()
        return float(np.dot(w_valid, block[valid]))
    
    def _build_rows_with_leads(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series
    ) -> Tuple[List, np.ndarray, List]:
        """
        Build training rows WITH LEADS for MIDAS estimation.
        
        For each observation:
        - Target: y_{t+1} (GDP of quarter t+1)
        - AR lags: y_t, y_{t-1}, ...
        - Daily block: lags from quarter t and before + leads from quarter t+1
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            # Target: next quarter's GDP
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            # Check AR lags availability
            if i < self.spec.p_y:
                continue
            
            # Collect AR lags (y_t, y_{t-1}, ...)
            y_lags = []
            for j in range(self.spec.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.spec.p_y:
                continue
            
            # Compute info date for target quarter
            info_date = self._compute_info_date(t_target)
            
            # Extract block with leads
            block = self._extract_block_with_leads(x_daily, t, info_date)
            
            # Require at least 50% valid data
            valid_count = (~np.isnan(block)).sum()
            total_m = self.spec.m + self.lead_days
            if valid_count < total_m * 0.5:
                continue
            
            rows.append((y_lags, block))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, x_daily: pd.Series, theta_init: float = -0.05) -> bool:
        """
        Fit the MIDAS model with leads using nonlinear least squares.
        """
        rows, y, _ = self._build_rows_with_leads(yq, x_daily)
        
        if len(rows) < 10:
            return False
        
        n_params = (1 if self.spec.add_const else 0) + self.spec.p_y + 2
        
        def unpack(b):
            pos = 0
            c = b[pos] if self.spec.add_const else 0.0
            pos += (1 if self.spec.add_const else 0)
            rho = b[pos:pos+self.spec.p_y]
            pos += self.spec.p_y
            beta = b[pos]
            theta = b[pos+1]
            return c, rho, beta, theta
        
        def residuals(b):
            c, rho, beta, theta = unpack(b)
            yhat = np.zeros(len(y))
            for i, (y_lags, block) in enumerate(rows):
                pred = c + float(np.dot(rho, np.array(y_lags)))
                pred += beta * self._midas_aggregate_extended(block, theta)
                yhat[i] = pred
            return y - yhat
        
        # Initialize
        b0 = np.zeros(n_params)
        pos = 0
        if self.spec.add_const:
            b0[pos] = np.mean(y)
            pos += 1
        b0[pos:pos+self.spec.p_y] = 0.3
        pos += self.spec.p_y
        b0[pos] = 0.0
        b0[pos+1] = theta_init
        
        # Bounds
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)
        lb[-1], ub[-1] = -1.0, 1.0
        
        try:
            res = least_squares(residuals, b0, bounds=(lb, ub), method="trf", max_nfev=1000)
            c, rho, beta, theta = unpack(res.x)
            self.params = {
                "const": float(c),
                "rho": rho.astype(float),
                "beta": float(beta),
                "theta": float(theta)
            }
            self.fitted = res.success
            return self.fitted
        except Exception:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts WITH LEADS.
        
        At each forecast origin t:
        1. Target = GDP of quarter t+h
        2. Info date = end of month 2 of quarter t+h
        3. Use daily data up to info_date (includes leads)
        4. Estimate model and forecast
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Compute info date for this target
            info_date = self._compute_info_date(target_date)
            
            # Fit on training data (up to train_end for GDP, up to info_date for daily)
            y_train = yq.loc[:train_end]
            
            # Create training model
            model = MidasModelWithLeads(self.spec, self.lead_months)
            if not model.fit(y_train, x_daily):
                continue
            
            # Forecast using leads
            y_lags = [float(yq.iloc[i-j]) for j in range(self.spec.p_y) if i-j >= 0]
            if len(y_lags) != self.spec.p_y:
                continue
            
            # Extract block with leads up to info_date
            block = self._extract_block_with_leads(x_daily, train_end, info_date)
            total_m = self.spec.m + self.lead_days
            if (~np.isnan(block)).sum() < total_m * 0.5:
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["beta"] * self._midas_aggregate_extended(block, model.params["theta"])
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "info_date": info_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "theta": model.params["theta"],
                "beta": model.params["beta"],
                "lead_days_used": self.lead_days,
                "total_m": total_m
            })
        
        if len(out) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# LAG SELECTION BY INFORMATION CRITERIA (AIC/BIC)
# Following Andreou et al. (2013) methodology
# =============================================================================

def compute_aic_bic(residuals: np.ndarray, n_params: int) -> Tuple[float, float]:
    """
    Compute AIC and BIC for model selection.
    
    AIC = n * log(RSS/n) + 2*k
    BIC = n * log(RSS/n) + k*log(n)
    
    where n = sample size, k = number of parameters, RSS = sum of squared residuals
    """
    n = len(residuals)
    rss = np.sum(residuals**2)
    
    if rss <= 0 or n <= n_params:
        return np.inf, np.inf
    
    log_likelihood = -n/2 * (1 + np.log(2*np.pi) + np.log(rss/n))
    
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)
    
    return aic, bic


def select_lag_order(
    yq: pd.Series,
    x_daily: pd.Series,
    max_p_y: int = 4,
    m: int = 63,
    h: int = 1,
    criterion: str = "BIC",
    verbose: bool = True
) -> Tuple[int, pd.DataFrame]:
    """
    Select optimal lag order p_y using information criteria (AIC or BIC).
    
    Following Andreou et al. (2013), the paper uses AIC for lag selection.
    BIC is also available as it penalizes complexity more heavily.
    
    Parameters
    ----------
    yq : pd.Series
        Quarterly GDP growth
    x_daily : pd.Series
        Daily factor or indicator series
    max_p_y : int
        Maximum number of AR lags to test (default: 4)
    m : int
        MIDAS window size in days
    h : int
        Forecast horizon
    criterion : str
        "AIC" or "BIC" (default: "BIC")
    verbose : bool
        Print selection results
        
    Returns
    -------
    optimal_p : int
        Optimal lag order
    results_df : pd.DataFrame
        AIC/BIC values for each lag order
    """
    results = []
    
    for p_y in range(1, max_p_y + 1):
        spec = MidasSpec(h=h, p_y=p_y, m=m, add_const=True)
        model = MidasModel(spec)
        
        # Fit model on full sample to compute information criteria
        rows, y, _ = model._build_rows(yq, x_daily)
        
        if len(rows) < 20:
            results.append({
                "p_y": p_y,
                "n_obs": len(rows),
                "n_params": np.nan,
                "RSS": np.nan,
                "AIC": np.inf,
                "BIC": np.inf,
            })
            continue
        
        if model.fit(yq, x_daily):
            # Compute residuals on training data
            residuals = []
            for i, (y_lags, block) in enumerate(rows):
                pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
                pred += model.params["beta"] * midas_aggregate(block, model.params["theta"])
                residuals.append(y[i] - pred)
            
            residuals = np.array(residuals)
            n_params = (1 + p_y + 2)  # const + p_y AR lags + beta + theta
            
            aic, bic = compute_aic_bic(residuals, n_params)
            rss = np.sum(residuals**2)
            
            results.append({
                "p_y": p_y,
                "n_obs": len(rows),
                "n_params": n_params,
                "RSS": rss,
                "AIC": aic,
                "BIC": bic,
            })
        else:
            results.append({
                "p_y": p_y,
                "n_obs": len(rows),
                "n_params": np.nan,
                "RSS": np.nan,
                "AIC": np.inf,
                "BIC": np.inf,
            })
    
    df = pd.DataFrame(results)
    
    # Select optimal lag
    if criterion.upper() == "AIC":
        optimal_p = int(df.loc[df["AIC"].idxmin(), "p_y"])
    else:  # BIC
        optimal_p = int(df.loc[df["BIC"].idxmin(), "p_y"])
    
    if verbose:
        print(f"LAG SELECTION ({criterion})")
        print("-" * 50)
        print(df[["p_y", "n_obs", "AIC", "BIC"]].to_string(index=False))
        print(f"\n→ Optimal p_y = {optimal_p} (by {criterion})")
        print("-" * 50)
    
    return optimal_p, df


def estimate_with_optimal_lags(
    yq: pd.Series,
    x_daily: pd.Series,
    start_oos: pd.Timestamp,
    max_p_y: int = 4,
    m: int = 63,
    h: int = 1,
    criterion: str = "BIC",
    min_train_obs: int = 20,
    verbose: bool = True
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Estimate MIDAS model with optimal lag order selected by AIC/BIC.
    
    Lag order is selected ONCE on the initial training sample (before OOS period)
    to avoid look-ahead bias.
    
    Parameters
    ----------
    yq : pd.Series
        Quarterly GDP growth
    x_daily : pd.Series
        Daily factor or indicator
    start_oos : pd.Timestamp
        Start of out-of-sample period
    max_p_y : int
        Maximum lags to consider
    m : int
        MIDAS window
    h : int
        Forecast horizon
    criterion : str
        "AIC" or "BIC"
    min_train_obs : int
        Minimum training observations
    verbose : bool
        Print results
        
    Returns
    -------
    fc : pd.DataFrame
        Forecast results
    optimal_p : int
        Selected lag order
    lag_selection_df : pd.DataFrame
        AIC/BIC for each lag order
    """
    # Use only pre-OOS data for lag selection to avoid look-ahead bias
    yq_train = yq[yq.index < start_oos]
    x_train = x_daily[x_daily.index < start_oos]
    
    if verbose:
        print(f"Selecting lags on training sample (before {start_oos.date()})")
    
    optimal_p, lag_df = select_lag_order(
        yq_train, x_train,
        max_p_y=max_p_y, m=m, h=h, criterion=criterion, verbose=verbose
    )
    
    # Estimate with optimal lags
    spec = MidasSpec(h=h, p_y=optimal_p, m=m, add_const=True)
    model = MidasModel(spec)
    fc = model.recursive_forecast(yq, x_daily, start_oos, min_train_obs=min_train_obs)
    
    if verbose and len(fc) > 0:
        print(f"\nMIDAS(p_y={optimal_p}): {len(fc)} forecasts, RMSFE = {rmsfe(fc):.4f}")
    
    return fc, optimal_p, lag_df


# =============================================================================
# MACRO INDICATORS ESTIMATION (Andreou et al. 2013)
# =============================================================================

# Default macro indicators available in Bloomberg
DEFAULT_MACRO_INDICATORS = {
    "ADS": "ADS BCI Index",      # Daily (Aruoba-Diebold-Scotti Business Conditions)
    "CFNAI": "CFNAI Index",       # Monthly (Chicago Fed National Activity Index)
    "PMI": "NAPMPMI Index",       # Monthly (ISM Manufacturing PMI)
}


def estimate_macro_midas(
    df_raw: pd.DataFrame,
    yq: pd.Series,
    start_oos: pd.Timestamp,
    rmsfe_ar1: float,
    macro_indicators: Dict[str, str] = None,
    min_train_obs: int = 20,
    verbose: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Estimate MIDAS models for macro indicators following Andreou et al. (2013).
    
    The paper uses monthly/daily macro indicators directly:
    - ADS: Daily → m=63 (1 quarter of daily data)
    - CFNAI, PMI: Monthly → m=3 (1 quarter of monthly data)
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw Bloomberg data with macro indicators
    yq : pd.Series
        Quarterly GDP growth target
    start_oos : pd.Timestamp
        Start of out-of-sample period
    rmsfe_ar1 : float
        RMSFE of AR(1) benchmark for comparison
    macro_indicators : dict, optional
        Dictionary {name: ticker} of macro indicators to use
    min_train_obs : int
        Minimum training observations required
    verbose : bool
        Print progress information
        
    Returns
    -------
    macro_forecasts : dict
        Dictionary of forecast DataFrames for each indicator
    macro_rmsfe : dict
        Dictionary of RMSFE values for each indicator
    """
    if macro_indicators is None:
        macro_indicators = DEFAULT_MACRO_INDICATORS
    
    macro_forecasts = {}
    macro_rmsfe = {}
    
    if verbose:
        print("MIDAS - Macro Indicators (Andreou et al. 2013)")
        print("=" * 60)
    
    for name, ticker in macro_indicators.items():
        if ticker not in df_raw.columns:
            if verbose:
                print(f"  ✗ {name}: Ticker '{ticker}' not found")
            continue
        
        # Extract and clean series
        macro_series = df_raw[ticker].dropna()
        
        if len(macro_series) < 100:
            if verbose:
                print(f"  ✗ {name}: Insufficient data ({len(macro_series)} obs)")
            continue
        
        # Determine frequency and adapt m
        # ADS is daily → m=63, CFNAI/PMI are monthly → resample to m=3
        if name == "ADS":
            spec_macro = MidasSpec(h=1, p_y=1, m=63, add_const=True)
            x_macro = macro_series
        else:
            # Monthly indicators: resample to end-of-month and use m=3
            x_macro = macro_series.resample('M').last().dropna()
            spec_macro = MidasSpec(h=1, p_y=1, m=3, add_const=True)
        
        # Estimate MIDAS model
        model_macro = MidasModel(spec_macro)
        fc_macro = model_macro.recursive_forecast(yq, x_macro, start_oos, min_train_obs=min_train_obs)
        
        if fc_macro is not None and len(fc_macro) > 0:
            macro_forecasts[name] = fc_macro
            macro_rmsfe[name] = rmsfe(fc_macro)
            
            if verbose:
                gain_vs_ar1 = (rmsfe_ar1 - macro_rmsfe[name]) / rmsfe_ar1 * 100
                freq = "daily, m=63" if name == "ADS" else "monthly, m=3"
                print(f"  ✓ {name} ({freq}): RMSFE = {macro_rmsfe[name]:.4f} | vs AR(1): {gain_vs_ar1:+.1f}%")
        else:
            if verbose:
                print(f"  ✗ {name}: Estimation failed")
    
    if verbose:
        print("=" * 60)
    
    return macro_forecasts, macro_rmsfe


# =============================================================================
# State-Dependent MIDAS Weights Extension
# θ_t = θ_0 + θ_1 * Z_t where Z_t is a stress indicator (e.g., VIX)
# =============================================================================

def exp_almon_weights_state_dep(m: int, theta0: float, theta1: float, z_t: float) -> np.ndarray:
    """
    Exponential Almon weights with state-dependent parameter.
    
    θ_t = θ_0 + θ_1 * Z_t
    
    Hypothesis: θ_1 < 0 implies that during stress (high Z_t), 
    the weighting shifts toward more recent observations.
    """
    theta_t = theta0 + theta1 * z_t
    return exp_almon_weights(m, theta_t)


def midas_aggregate_state_dep(block: np.ndarray, theta0: float, theta1: float, z_t: float) -> float:
    """
    Weighted aggregation using state-dependent weights.
    """
    m = len(block)
    w = exp_almon_weights_state_dep(m, theta0, theta1, z_t)
    valid_mask = ~np.isnan(block)
    if valid_mask.sum() == 0:
        return 0.0
    w_valid = w[valid_mask]
    w_valid = w_valid / w_valid.sum()
    return float(np.dot(w_valid, block[valid_mask]))


class MidasModelStateDep:
    """
    -> It was our first idea of extension, but not used in the notebook. (didn't lead to any improvement but only more complexity)
    State-Dependent ADL-MIDAS Model Extension.
    
    Model:
        y_{t+h} = α + ρ * y_t + β * Σ B(k; θ_t) * x_{t-k} + ε_{t+h}
    
    Where:
        θ_t = θ_0 + θ_1 * Z_t
        Z_t = stress indicator (e.g., standardized VIX)
    
    Hypothesis:
        θ_1 < 0: In stressed periods, more weight on recent data
    
    Reference:
        Extension of Andreou, Ghysels, Kourtellos (2013)
    """
    
    def __init__(self, spec: MidasSpec):
        self.spec = spec
        self.fitted = False
        self.params = None
    
    def _extract_last_m(self, x_daily: pd.Series, q_date: pd.Timestamp) -> np.ndarray:
        """Extract last m daily observations before quarter end."""
        mask = x_daily.index <= q_date
        sub = x_daily.loc[mask].tail(self.spec.m)
        arr = np.full(self.spec.m, np.nan)
        if len(sub) > 0:
            arr[-len(sub):] = sub.values
        return arr
    
    def _get_stress_indicator(self, z_daily: pd.Series, q_date: pd.Timestamp) -> float:
        """
        Get average stress indicator for the quarter.
        Returns standardized value (z-score).
        """
        mask = z_daily.index <= q_date
        sub = z_daily.loc[mask].tail(self.spec.m)
        if len(sub) == 0:
            return 0.0
        return float(sub.mean())
    
    def _build_rows(self, yq: pd.Series, x_daily: pd.Series, z_daily: pd.Series):
        """Build observation rows including stress indicator."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates, z_vals = [], [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.spec.p_y:
                continue
            
            y_lags = []
            for j in range(self.spec.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.spec.p_y:
                continue
            
            block = self._extract_last_m(x_daily, t)
            valid_count = (~np.isnan(block)).sum()
            if valid_count < self.spec.m * 0.5:
                continue
            
            z_t = self._get_stress_indicator(z_daily, t)
            
            rows.append((y_lags, block))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
            z_vals.append(z_t)
        
        return rows, np.array(y_out), target_dates, np.array(z_vals)
    
    def fit(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        z_daily: pd.Series,
        theta0_init: float = -0.05,
        theta1_init: float = -0.01
    ) -> bool:
        """
        Fit the State-Dependent MIDAS model using NLS.
        
        Parameters:
            yq: Quarterly GDP growth
            x_daily: Daily factor/regressor
            z_daily: Daily stress indicator (e.g., VIX) - should be standardized
        """
        rows, y, _, z_vals = self._build_rows(yq, x_daily, z_daily)
        
        if len(rows) < 10:
            return False
        
        # Parameters: const + rho + beta + theta0 + theta1
        n_params = (1 if self.spec.add_const else 0) + self.spec.p_y + 3
        
        def unpack(b):
            pos = 0
            c = b[pos] if self.spec.add_const else 0.0
            pos += (1 if self.spec.add_const else 0)
            rho = b[pos:pos+self.spec.p_y]
            pos += self.spec.p_y
            beta = b[pos]
            theta0 = b[pos+1]
            theta1 = b[pos+2]
            return c, rho, beta, theta0, theta1
        
        def residuals(b):
            c, rho, beta, theta0, theta1 = unpack(b)
            yhat = np.zeros(len(y))
            for i, (y_lags, block) in enumerate(rows):
                pred = c + float(np.dot(rho, np.array(y_lags)))
                pred += beta * midas_aggregate_state_dep(block, theta0, theta1, z_vals[i])
                yhat[i] = pred
            return y - yhat
        
        # Initialize
        b0 = np.zeros(n_params)
        pos = 0
        if self.spec.add_const:
            b0[pos] = np.mean(y)
            pos += 1
        b0[pos:pos+self.spec.p_y] = 0.3
        pos += self.spec.p_y
        b0[pos] = 0.0      # beta
        b0[pos+1] = theta0_init
        b0[pos+2] = theta1_init
        
        # Bounds: theta0, theta1 in [-2, 2]
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)
        lb[-2], ub[-2] = -2.0, 2.0   # theta0
        lb[-1], ub[-1] = -2.0, 2.0   # theta1
        
        try:
            res = least_squares(residuals, b0, bounds=(lb, ub), method="trf", max_nfev=2000)
            c, rho, beta, theta0, theta1 = unpack(res.x)
            self.params = {
                "const": float(c),
                "rho": rho.astype(float),
                "beta": float(beta),
                "theta0": float(theta0),
                "theta1": float(theta1)
            }
            self.fitted = res.success
            return self.fitted
        except Exception:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series,
        z_daily: pd.Series,
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts with state-dependent weights.
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            y_train = yq.loc[:train_end]
            rows, y_vec, _, _ = self._build_rows(y_train, x_daily, z_daily)
            
            if len(rows) < min_train_obs:
                continue
            
            # Estimate
            model = MidasModelStateDep(self.spec)
            if not model.fit(y_train, x_daily, z_daily):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.spec.p_y) if i-j >= 0]
            if len(y_lags) != self.spec.p_y:
                continue
            
            block = self._extract_last_m(x_daily, train_end)
            if (~np.isnan(block)).sum() < self.spec.m * 0.5:
                continue
            
            z_t = self._get_stress_indicator(z_daily, train_end)
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["beta"] * midas_aggregate_state_dep(
                block, model.params["theta0"], model.params["theta1"], z_t
            )
            
            # Compute effective theta
            theta_t = model.params["theta0"] + model.params["theta1"] * z_t
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "theta0": model.params["theta0"],
                "theta1": model.params["theta1"],
                "z_t": z_t,
                "theta_t": theta_t,
                "beta": model.params["beta"],
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=[
                "forecast_origin", "target_date", "y_true", "y_pred", 
                "fe", "se", "theta0", "theta1", "z_t", "theta_t"
            ])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# FADL-MIDAS WITH LEADS ON MACRO AND DAILY (Table 3 - Full Model)
# FADL-MIDAS(J_CFNAI^M = 1, J_X^D = 2) - Andreou et al. (2013)
# =============================================================================

class FADLMidasWithLeads:
    """
    FADL-MIDAS model with leads on BOTH monthly macro factor AND daily factor.
    
    This is the FULL model from Table 3 of Andreou et al. (2013):
    - FADL-MIDAS(J_CFNAI^M = 1, J_X^D = 2)
    - FADL-MIDAS(J_NAPMNOI^M = 2, J_X^D = 2)
    
    Equation (from paper's equation 2.7 extended):
    y_{t+h} = c + ρ*y_t + γ*F_{t+J_M}^M + β*B(L^{1/m}; θ)*X_{t+J_D}^D + ε_{t+h}
    
    where:
    - F_{t+J_M}^M = monthly macro factor with J_M months of leads
    - X_{t+J_D}^D = daily factor with J_D months of leads
    - B(L^{1/m}; θ) = MIDAS weighting polynomial
    
    Parameters:
    -----------
    spec : MidasSpec
        Base MIDAS specification (h, p_y, m)
    lead_months_macro : int
        Number of months of leads for monthly macro factor (J_M = 1 or 2)
    lead_months_daily : int  
        Number of months of leads for daily factor (J_D = 0, 1, or 2)
    """
    
    def __init__(
        self, 
        spec: MidasSpec, 
        lead_months_macro: int = 1,  # J_M
        lead_months_daily: int = 2   # J_D
    ):
        """
        Initialize FADL-MIDAS model with leads on both macro and daily.
        
        Args:
            spec: MidasSpec with h, p_y, m parameters
            lead_months_macro: Months of leads for monthly macro (paper: 1 for CFNAI, 2 for NAPMNOI)
            lead_months_daily: Months of leads for daily factor (paper: 0, 1, or 2)
        """
        self.spec = spec
        self.lead_months_macro = lead_months_macro
        self.lead_months_daily = lead_months_daily
        self.lead_days = lead_months_daily * 21  # ~21 trading days per month
        self.params = None
        self.fitted = False
    
    def _compute_info_date_daily(self, target_quarter_end: pd.Timestamp) -> pd.Timestamp:
        """
        Compute info date for daily factor with leads.
        
        Info date = end of month (lead_months_daily) within the target quarter.
        For J_D=2: end of month 2 of target quarter
        For J_D=0: end of previous quarter (no leads)
        """
        if self.lead_months_daily == 0:
            # No leads - use previous quarter end
            return target_quarter_end - pd.DateOffset(months=3) + pd.offsets.QuarterEnd(0)
        
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        info_month = quarter_start + pd.DateOffset(months=self.lead_months_daily - 1)
        return info_month + pd.offsets.MonthEnd(0)
    
    def _compute_info_date_macro(self, target_quarter_end: pd.Timestamp) -> pd.Timestamp:
        """
        Compute info date for monthly macro factor with leads.
        
        Info date for monthly data depends on J_M.
        For J_M=1: end of month 1 of target quarter
        For J_M=2: end of month 2 of target quarter
        """
        if self.lead_months_macro == 0:
            return target_quarter_end - pd.DateOffset(months=3) + pd.offsets.QuarterEnd(0)
        
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        info_month = quarter_start + pd.DateOffset(months=self.lead_months_macro - 1)
        return info_month + pd.offsets.MonthEnd(0)
    
    def _extract_block_with_leads(
        self, 
        x_daily: pd.Series, 
        last_known_quarter_end: pd.Timestamp,
        info_date: pd.Timestamp
    ) -> np.ndarray:
        """Extract daily block including leads up to info_date."""
        total_m = self.spec.m + self.lead_days
        s = x_daily.loc[:info_date].dropna()
        
        arr = np.full(total_m, np.nan)
        if len(s) == 0:
            return arr
        
        take = s.iloc[-total_m:].values
        arr[-len(take):] = take
        return arr
    
    def _extract_block_no_leads(
        self, 
        x_daily: pd.Series, 
        quarter_end: pd.Timestamp
    ) -> np.ndarray:
        """Extract standard daily block (no leads)."""
        s = x_daily.loc[:quarter_end].dropna()
        arr = np.full(self.spec.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.spec.m:].values
        arr[-len(take):] = take
        return arr
    
    def _get_macro_factor_with_leads(
        self, 
        f_monthly: pd.Series, 
        info_date: pd.Timestamp
    ) -> float:
        """
        Get macro factor value with leads.
        
        The macro factor (e.g., CFNAI) is monthly, so we take the most recent
        value available up to info_date.
        """
        s = f_monthly.loc[:info_date].dropna()
        if len(s) == 0:
            return np.nan
        return float(s.iloc[-1])
    
    def _exp_almon_weights(self, total_m: int, theta: float) -> np.ndarray:
        """Compute exponential Almon weights."""
        k = np.arange(total_m)
        theta = np.clip(theta, -1.0, 1.0)
        a = np.exp(theta * k)
        return a / a.sum()
    
    def _midas_aggregate(self, block: np.ndarray, theta: float) -> float:
        """Aggregate block using MIDAS weights."""
        total_m = len(block)
        w = self._exp_almon_weights(total_m, theta)
        
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return 0.0
        
        w_valid = w[valid]
        w_valid = w_valid / w_valid.sum()
        return float(np.dot(w_valid, block[valid]))
    
    def _build_rows(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series,
        f_monthly: pd.Series
    ) -> Tuple[List, np.ndarray, List]:
        """
        Build training rows for FADL-MIDAS with leads.
        
        Each row contains:
        - AR lags: y_t, y_{t-1}, ...
        - Macro factor with leads: F_{t+J_M}
        - Daily block with leads for MIDAS aggregation
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            # Check AR lags availability
            if i < self.spec.p_y:
                continue
            
            # Collect AR lags
            y_lags = []
            for j in range(self.spec.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.spec.p_y:
                continue
            
            # Info dates for daily and macro
            info_date_daily = self._compute_info_date_daily(t_target)
            info_date_macro = self._compute_info_date_macro(t_target)
            
            # Extract daily block with leads
            if self.lead_months_daily > 0:
                block = self._extract_block_with_leads(x_daily, t, info_date_daily)
                total_m = self.spec.m + self.lead_days
            else:
                block = self._extract_block_no_leads(x_daily, t)
                total_m = self.spec.m
            
            valid_count = (~np.isnan(block)).sum()
            if valid_count < total_m * 0.5:
                continue
            
            # Get macro factor with leads
            f_macro = self._get_macro_factor_with_leads(f_monthly, info_date_macro)
            if np.isnan(f_macro):
                continue
            
            rows.append((y_lags, block, f_macro))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series,
        f_monthly: pd.Series,
        theta_init: float = -0.05
    ) -> bool:
        """
        Fit FADL-MIDAS with leads using NLS.
        
        Parameters:
            yq: Quarterly GDP growth
            x_daily: Daily factor (e.g., PCA factor)
            f_monthly: Monthly macro factor (e.g., CFNAI)
            theta_init: Initial value for MIDAS theta
        """
        rows, y, _ = self._build_rows(yq, x_daily, f_monthly)
        
        if len(rows) < 10:
            return False
        
        # Parameters: const + p_y AR lags + gamma (macro) + beta (daily) + theta
        n_params = (1 if self.spec.add_const else 0) + self.spec.p_y + 3
        
        def unpack(b):
            pos = 0
            c = b[pos] if self.spec.add_const else 0.0
            pos += (1 if self.spec.add_const else 0)
            rho = b[pos:pos+self.spec.p_y]
            pos += self.spec.p_y
            gamma = b[pos]      # coef for macro factor
            beta = b[pos+1]     # coef for daily MIDAS
            theta = b[pos+2]    # MIDAS weight parameter
            return c, rho, gamma, beta, theta
        
        def residuals(b):
            c, rho, gamma, beta, theta = unpack(b)
            yhat = np.zeros(len(y))
            for i, (y_lags, block, f_macro) in enumerate(rows):
                pred = c + float(np.dot(rho, np.array(y_lags)))
                pred += gamma * f_macro
                pred += beta * self._midas_aggregate(block, theta)
                yhat[i] = pred
            return y - yhat
        
        # Initialize
        b0 = np.zeros(n_params)
        pos = 0
        if self.spec.add_const:
            b0[pos] = np.mean(y)
            pos += 1
        b0[pos:pos+self.spec.p_y] = 0.3
        pos += self.spec.p_y
        b0[pos] = 0.0          # gamma
        b0[pos+1] = 0.0        # beta
        b0[pos+2] = theta_init # theta
        
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)
        lb[-1], ub[-1] = -1.0, 1.0  # theta bounds
        
        try:
            res = least_squares(residuals, b0, bounds=(lb, ub), method="trf", max_nfev=2000)
            c, rho, gamma, beta, theta = unpack(res.x)
            self.params = {
                "const": float(c),
                "rho": rho.astype(float),
                "gamma": float(gamma),
                "beta": float(beta),
                "theta": float(theta)
            }
            self.fitted = res.success
            return self.fitted
        except Exception:
            return False
    
    def predict(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series,
        f_monthly: pd.Series,
        forecast_origin: pd.Timestamp
    ) -> float:
        """
        Make a single out-of-sample forecast.
        
        Args:
            yq: Historical GDP series (up to and including forecast_origin)
            x_daily: Daily factor series
            f_monthly: Monthly macro factor series
            forecast_origin: Date of the last known GDP (quarter end)
        
        Returns:
            Forecast for h quarters ahead
        """
        if not self.fitted or self.params is None:
            return np.nan
        
        yq = yq.loc[:forecast_origin].dropna()
        if len(yq) < self.spec.p_y:
            return np.nan
        
        # AR lags
        y_lags = yq.iloc[-self.spec.p_y:].values[::-1]
        
        # Target quarter end
        target_quarter_end = forecast_origin + pd.DateOffset(months=3 * self.spec.h)
        target_quarter_end = target_quarter_end + pd.offsets.QuarterEnd(0)
        
        # Info dates
        info_date_daily = self._compute_info_date_daily(target_quarter_end)
        info_date_macro = self._compute_info_date_macro(target_quarter_end)
        
        # Daily block with leads
        if self.lead_months_daily > 0:
            block = self._extract_block_with_leads(x_daily, forecast_origin, info_date_daily)
        else:
            block = self._extract_block_no_leads(x_daily, forecast_origin)
        
        # Macro factor with leads
        f_macro = self._get_macro_factor_with_leads(f_monthly, info_date_macro)
        
        if np.isnan(block).all() or np.isnan(f_macro):
            return np.nan
        
        p = self.params
        pred = p["const"] + float(np.dot(p["rho"], y_lags))
        pred += p["gamma"] * f_macro
        pred += p["beta"] * self._midas_aggregate(block, p["theta"])
        
        return pred
    
    def rolling_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series,
        f_monthly: pd.Series,
        start_oos: pd.Timestamp,
        expanding: bool = True
    ) -> pd.DataFrame:
        """
        Perform rolling/expanding window out-of-sample forecasts.
        
        Args:
            yq: Full quarterly GDP series
            x_daily: Full daily factor series
            f_monthly: Full monthly macro factor series
            start_oos: First out-of-sample date
            expanding: If True, use expanding window; if False, use rolling window
        
        Returns:
            DataFrame with columns: forecast_origin, target_date, y_true, y_pred, fe, se
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        oos_dates = [d for d in q_dates if d >= start_oos]
        out = []
        
        for fo in oos_dates:
            # Target date
            fo_idx = q_dates.index(fo)
            target_idx = fo_idx + self.spec.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Training sample
            if expanding:
                train_end = fo
            else:
                # Rolling window of ~80 quarters
                window_size = 80
                start_idx = max(0, fo_idx - window_size)
                train_end = q_dates[fo_idx]
            
            yq_train = yq.loc[:train_end]
            
            # Refit model
            success = self.fit(yq_train, x_daily, f_monthly)
            if not success:
                continue
            
            # Forecast
            yhat = self.predict(yq, x_daily, f_monthly, fo)
            if np.isnan(yhat):
                continue
            
            y_true = float(yq.loc[target_date])
            
            out.append({
                "forecast_origin": fo,
                "target_date": target_date,
                "y_true": y_true,
                "y_pred": yhat,
                "theta": self.params["theta"],
                "gamma": self.params["gamma"],
                "beta": self.params["beta"]
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=[
                "forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"
            ])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df





# =============================================================================
# EXTENSION: MIDAS MODEL WITH TWO THETA (SEPARATE FOR LAGS AND LEADS)
# =============================================================================

class MidasModelTwoTheta:
    """
    ADL-MIDAS Model with SEPARATE theta parameters for lags and leads.
    
    Extension of Andreou et al. (2013):
    - θ_lag: exponential Almon parameter for the LAG block (past data)
    - θ_lead: exponential Almon parameter for the LEAD block (nowcast data)
    
    Model specification:
        y_{t+h} = α + ρ·y_t + β_lag·Σ B(k;θ_lag)·x_{t-k} + β_lead·Σ B(j;θ_lead)·x_{t+j} + ε
    
    This allows different weighting schemes for historical vs. nowcast information.
    
    Parameters:
    -----------
    spec : MidasSpec
        Model specification (h, p_y, m, add_const)
    lead_months : int
        Number of months of leads (1 or 2)
    """
    
    def __init__(self, spec: MidasSpec, lead_months: int = 2):
        """Initialize Two-Theta MIDAS model."""
        self.spec = spec
        self.lead_months = lead_months
        self.lead_days = lead_months * 21  # ~21 trading days per month
        self.params = None
        self.fitted = False
    
    def _compute_info_date(self, target_quarter_end: pd.Timestamp) -> pd.Timestamp:
        """Compute information date for forecasting a given quarter."""
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        info_month = quarter_start + pd.DateOffset(months=self.lead_months - 1)
        info_date = info_month + pd.offsets.MonthEnd(0)
        return info_date
    
    def _extract_lag_block(self, x_daily: pd.Series, end_date: pd.Timestamp) -> np.ndarray:
        """Extract the LAG block (m observations up to end_date)."""
        s = x_daily.loc[:end_date].dropna()
        arr = np.full(self.spec.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.spec.m:].values
        arr[-len(take):] = take
        return arr
    
    def _extract_lead_block(
        self, 
        x_daily: pd.Series, 
        last_quarter_end: pd.Timestamp,
        info_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Extract the LEAD block (data from target quarter up to info_date).
        
        Leads are data AFTER last_quarter_end and up to info_date.
        """
        # Get data after last_quarter_end and up to info_date
        mask = (x_daily.index > last_quarter_end) & (x_daily.index <= info_date)
        s = x_daily.loc[mask].dropna()
        
        arr = np.full(self.lead_days, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.lead_days:].values
        arr[-len(take):] = take
        return arr
    
    def _exp_almon_weights(self, m: int, theta: float) -> np.ndarray:
        """Compute normalized exponential Almon weights."""
        k = np.arange(m - 1, -1, -1)  # k = m-1 (oldest) to 0 (most recent)
        theta = np.clip(theta, -1.0, 1.0)
        a = np.exp(theta * k)
        return a / a.sum()
    
    def _midas_aggregate(self, block: np.ndarray, theta: float) -> float:
        """Aggregate a block using MIDAS weights."""
        m = len(block)
        w = self._exp_almon_weights(m, theta)
        
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return 0.0
        
        w_valid = w[valid]
        w_valid = w_valid / w_valid.sum()
        return float(np.dot(w_valid, block[valid]))
    
    def _build_rows(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series
    ) -> Tuple[List, np.ndarray, List]:
        """Build training rows with separate lag and lead blocks."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.spec.p_y:
                continue
            
            # AR lags
            y_lags = []
            for j in range(self.spec.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.spec.p_y:
                continue
            
            # Info date for target quarter
            info_date = self._compute_info_date(t_target)
            
            # Separate blocks
            lag_block = self._extract_lag_block(x_daily, t)
            lead_block = self._extract_lead_block(x_daily, t, info_date)
            
            # Require sufficient data in both blocks
            lag_valid = (~np.isnan(lag_block)).sum()
            lead_valid = (~np.isnan(lead_block)).sum()
            
            # Relaxed constraints for recent data
            if lag_valid < self.spec.m * 0.3 or lead_valid < max(1, self.lead_days * 0.1):
                continue
            
            rows.append((y_lags, lag_block, lead_block))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        theta_lag_init: float = -0.05,
        theta_lead_init: float = -0.05,
        min_obs: int = 6
    ) -> bool:
        """
        Fit the Two-Theta MIDAS model using nonlinear least squares.
        
        Parameters to estimate:
        - const (α)
        - rho (ρ, AR coefficient)
        - beta_lag (coefficient on lag MIDAS aggregate)
        - beta_lead (coefficient on lead MIDAS aggregate)
        - theta_lag (MIDAS weight parameter for lags)
        - theta_lead (MIDAS weight parameter for leads)
        """
        rows, y, _ = self._build_rows(yq, x_daily)
        
        if len(rows) < min_obs:
            return False
        
        # Parameters: const + p_y rhos + beta_lag + beta_lead + theta_lag + theta_lead
        n_params = (1 if self.spec.add_const else 0) + self.spec.p_y + 4
        
        def unpack(b):
            pos = 0
            c = b[pos] if self.spec.add_const else 0.0
            pos += (1 if self.spec.add_const else 0)
            rho = b[pos:pos+self.spec.p_y]
            pos += self.spec.p_y
            beta_lag = b[pos]
            beta_lead = b[pos+1]
            theta_lag = b[pos+2]
            theta_lead = b[pos+3]
            return c, rho, beta_lag, beta_lead, theta_lag, theta_lead
        
        def residuals(b):
            c, rho, beta_lag, beta_lead, theta_lag, theta_lead = unpack(b)
            yhat = np.zeros(len(y))
            for i, (y_lags, lag_block, lead_block) in enumerate(rows):
                pred = c + float(np.dot(rho, np.array(y_lags)))
                pred += beta_lag * self._midas_aggregate(lag_block, theta_lag)
                pred += beta_lead * self._midas_aggregate(lead_block, theta_lead)
                yhat[i] = pred
            return y - yhat
        
        # Initialize
        b0 = np.zeros(n_params)
        pos = 0
        if self.spec.add_const:
            b0[pos] = np.mean(y)
            pos += 1
        b0[pos:pos+self.spec.p_y] = 0.3
        pos += self.spec.p_y
        b0[pos] = 0.0      # beta_lag
        b0[pos+1] = 0.0    # beta_lead
        b0[pos+2] = theta_lag_init
        b0[pos+3] = theta_lead_init
        
        # Bounds: theta in [-1, 1]
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params, np.inf)
        lb[-2], ub[-2] = -1.0, 1.0  # theta_lag
        lb[-1], ub[-1] = -1.0, 1.0  # theta_lead
        
        try:
            res = least_squares(residuals, b0, bounds=(lb, ub), method="trf", max_nfev=2000)
            c, rho, beta_lag, beta_lead, theta_lag, theta_lead = unpack(res.x)
            self.params = {
                "const": float(c),
                "rho": rho.astype(float),
                "beta_lag": float(beta_lag),
                "beta_lead": float(beta_lead),
                "theta_lag": float(theta_lag),
                "theta_lead": float(theta_lead)
            }
            # Accept even if not fully converged, as long as params are valid
            self.fitted = True  # Changed: always accept if no exception
            return self.fitted
        except Exception as e:
            print(f"Fitting error: {e}")
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts with Two-Theta MIDAS.
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.spec.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            info_date = self._compute_info_date(target_date)
            
            # Fit on training data
            y_train = yq.loc[:train_end]
            
            model = MidasModelTwoTheta(self.spec, self.lead_months)
            if not model.fit(y_train, x_daily):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.spec.p_y) if i-j >= 0]
            if len(y_lags) != self.spec.p_y:
                continue
            
            lag_block = self._extract_lag_block(x_daily, train_end)
            lead_block = self._extract_lead_block(x_daily, train_end, info_date)
            
            # Relaxed constraints for recent data
            if (~np.isnan(lag_block)).sum() < self.spec.m * 0.3:
                continue
            if (~np.isnan(lead_block)).sum() < max(1, self.lead_days * 0.1):
                continue
            
            pred = model.params["const"]
            pred += float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["beta_lag"] * self._midas_aggregate(lag_block, model.params["theta_lag"])
            pred += model.params["beta_lead"] * self._midas_aggregate(lead_block, model.params["theta_lead"])
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "info_date": info_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "theta_lag": model.params["theta_lag"],
                "theta_lead": model.params["theta_lead"],
                "beta_lag": model.params["beta_lag"],
                "beta_lead": model.params["beta_lead"],
                "lead_days": self.lead_days
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=[
                "forecast_origin", "target_date", "y_true", "y_pred", 
                "theta_lag", "theta_lead", "beta_lag", "beta_lead", "fe", "se"
            ])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df



if __name__ == "__main__":
    print("=" * 60)