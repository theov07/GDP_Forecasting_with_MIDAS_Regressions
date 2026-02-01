"""
Benchmark Models Module
=======================

Benchmark models for GDP forecasting replication of Andreou, Ghysels, Kourtellos (2013).

Models implemented (Table 2 of paper):
- AR: Simple AR(p) benchmark
- ADL: Autoregressive Distributed Lag with FLAT aggregation (simple average)
- FAR: Factor AR (AR with macro factor like CFNAI)
- FADL: Factor ADL with flat aggregation
- Random Walk: Naive forecast

Also includes:
- Diebold-Mariano test for equal predictive ability
- MSFE-weighted forecast combination
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import stats


class RandomWalkModel:
    """
    Random Walk benchmark model for quarterly GDP forecasting.
    Model: ŷ_{t+h} = y_t (naive forecast: last observation)
    
    This is the simplest possible benchmark and represents the "no change" forecast.
    """
    
    def __init__(self, h: int = 1):
        """Initialize Random Walk model."""
        self.h = h
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        start_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts.
        Forecast: ŷ_{t+h} = y_t
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Random Walk: forecast = last observation
            y_last = float(yq.iloc[i])
            pred = y_last
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": pred,
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


class AR1Model:
    """
    Simple AR(1) benchmark model for quarterly GDP forecasting.
    Model: y_{t+h} = c + ρ·y_t + ε
    """
    
    def __init__(self, h: int = 1):
        """
        Initialize AR(1) model.
        """
        self.h = h
        self.const = None
        self.rho = None
        self.fitted = False
    
    def fit(self, yq: pd.Series) -> bool:
        """
        Fit AR(1) model using OLS.
        """
        y = yq.dropna().values
        
        if len(y) < 10:
            return False
        
        # OLS: y_t = c + rho * y_{t-1}
        Y = y[1:]
        X = np.column_stack([np.ones(len(Y)), y[:-1]])
        
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            self.const, self.rho = beta[0], beta[1]
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 10
    ) -> pd.DataFrame:
        """
        Generate recursive out-of-sample forecasts.
        """
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Fit on training data
            y_train = yq.loc[:train_end].values
            
            if len(y_train) < min_train_obs:
                continue
            
            # OLS estimation
            Y = y_train[1:]
            X = np.column_stack([np.ones(len(Y)), y_train[:-1]])
            
            try:
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                c, rho = beta[0], beta[1]
            except:
                continue
            
            # Forecast
            y_last = yq.iloc[i]
            pred = c + rho * y_last
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# ADL MODEL WITH FLAT AGGREGATION (Andreou et al. 2013, Table 2)
# =============================================================================

class ADLModel:
    """
    Autoregressive Distributed Lag model with FLAT (simple average) aggregation.
    
    Model: y_{t+h} = c + ρ·y_t + β·x̄_t + ε
    
    Where x̄_t = (1/m) Σ_{k=0}^{m-1} x_{t-k} (simple average, no MIDAS weights)
    
    This is the benchmark against which MIDAS weighting is compared.
    The key difference from ADL-MIDAS: uniform weights instead of exponential Almon.
    """
    
    def __init__(self, h: int = 1, p_y: int = 1, m: int = 63):
        """
        Initialize ADL model with flat aggregation.
        
        Parameters
        ----------
        h : int
            Forecast horizon in quarters
        p_y : int
            Number of AR lags
        m : int
            Number of daily observations to aggregate (63 = 1 quarter)
        """
        self.h = h
        self.p_y = p_y
        self.m = m
        self.params = None
        self.fitted = False
    
    def _flat_aggregate(self, block: np.ndarray) -> float:
        """Simple average aggregation (flat weights)."""
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return np.nan
        return float(np.mean(block[valid]))
    
    def _extract_last_m(self, series: pd.Series, end_date: pd.Timestamp) -> np.ndarray:
        """Extract the last m observations up to end_date."""
        s = series.loc[:end_date].dropna()
        arr = np.full(self.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.m:].values
        arr[-len(take):] = take
        return arr
    
    def _build_rows(self, yq: pd.Series, x_daily: pd.Series) -> Tuple[list, np.ndarray, list]:
        """Build training rows for ADL estimation."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.p_y:
                continue
            
            # Collect AR lags
            y_lags = []
            for j in range(self.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.p_y:
                continue
            
            # Extract daily block and compute flat average
            block = self._extract_last_m(x_daily, t)
            valid_count = (~np.isnan(block)).sum()
            if valid_count < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            rows.append((y_lags, x_agg))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, x_daily: pd.Series) -> bool:
        """Fit ADL model with flat aggregation using OLS."""
        rows, y, _ = self._build_rows(yq, x_daily)
        
        if len(rows) < 10:
            return False
        
        # Build design matrix: [1, y_lags, x_agg]
        n = len(rows)
        n_cols = 1 + self.p_y + 1  # const + AR lags + x_agg
        X = np.zeros((n, n_cols))
        
        for i, (y_lags, x_agg) in enumerate(rows):
            X[i, 0] = 1.0  # constant
            X[i, 1:1+self.p_y] = y_lags
            X[i, -1] = x_agg
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.params = {
                "const": beta[0],
                "rho": beta[1:1+self.p_y],
                "beta": beta[-1]
            }
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """Generate recursive out-of-sample forecasts with flat aggregation."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Fit on training data
            y_train = yq.loc[:train_end]
            rows, y_vec, _ = self._build_rows(y_train, x_daily)
            
            if len(rows) < min_train_obs:
                continue
            
            # Estimate
            model = ADLModel(h=self.h, p_y=self.p_y, m=self.m)
            if not model.fit(y_train, x_daily):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.p_y) if i-j >= 0]
            if len(y_lags) != self.p_y:
                continue
            
            block = self._extract_last_m(x_daily, train_end)
            if (~np.isnan(block)).sum() < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["beta"] * x_agg
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "beta": model.params["beta"],
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# FAR MODEL: Factor AR (AR with macro factor) - Andreou et al. 2013, Table 2
# =============================================================================

class FARModel:
    """
    Factor AR model: AR + quarterly macro factor (like CFNAI or S&W factor).
    
    Model: y_{t+h} = c + ρ·y_t + γ·F_t + ε
    
    Where F_t is a quarterly factor (e.g., CFNAI averaged to quarterly).
    
    This is the benchmark for testing whether financial data adds value
    beyond macro indicators.
    """
    
    def __init__(self, h: int = 1, p_y: int = 1):
        """
        Initialize FAR model.
        
        Parameters
        ----------
        h : int
            Forecast horizon
        p_y : int
            Number of AR lags
        """
        self.h = h
        self.p_y = p_y
        self.params = None
        self.fitted = False
    
    def _build_rows(self, yq: pd.Series, fq: pd.Series) -> Tuple[list, np.ndarray, list]:
        """Build training rows with quarterly factor."""
        yq = yq.dropna().sort_index()
        fq = fq.dropna().sort_index()
        
        # Align indices
        common_idx = yq.index.intersection(fq.index)
        yq = yq.loc[common_idx]
        fq = fq.loc[common_idx]
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.p_y:
                continue
            
            # Collect AR lags
            y_lags = []
            for j in range(self.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.p_y:
                continue
            
            # Factor value at t
            f_t = float(fq.iloc[i])
            if np.isnan(f_t):
                continue
            
            rows.append((y_lags, f_t))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, fq: pd.Series) -> bool:
        """Fit FAR model using OLS."""
        rows, y, _ = self._build_rows(yq, fq)
        
        if len(rows) < 10:
            return False
        
        n = len(rows)
        n_cols = 1 + self.p_y + 1  # const + AR lags + factor
        X = np.zeros((n, n_cols))
        
        for i, (y_lags, f_t) in enumerate(rows):
            X[i, 0] = 1.0
            X[i, 1:1+self.p_y] = y_lags
            X[i, -1] = f_t
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.params = {
                "const": beta[0],
                "rho": beta[1:1+self.p_y],
                "gamma": beta[-1]
            }
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        fq: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """Generate recursive OOS forecasts."""
        yq = yq.dropna().sort_index()
        fq = fq.dropna().sort_index()
        
        common_idx = yq.index.intersection(fq.index)
        yq = yq.loc[common_idx]
        fq = fq.loc[common_idx]
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            # Fit on training data
            y_train = yq.loc[:train_end]
            f_train = fq.loc[:train_end]
            rows, y_vec, _ = self._build_rows(y_train, f_train)
            
            if len(rows) < min_train_obs:
                continue
            
            model = FARModel(h=self.h, p_y=self.p_y)
            if not model.fit(y_train, f_train):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.p_y) if i-j >= 0]
            if len(y_lags) != self.p_y:
                continue
            
            f_t = float(fq.iloc[i])
            if np.isnan(f_t):
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["gamma"] * f_t
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "gamma": model.params["gamma"],
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# FADL MODEL: Factor ADL with flat aggregation - Andreou et al. 2013, Table 2
# =============================================================================

class FADLModel:
    """
    Factor ADL model: AR + quarterly factor + daily data (flat aggregation).
    
    Model: y_{t+h} = c + ρ·y_t + γ·F_t + β·x̄_t + ε
    
    Combines macro factor (like CFNAI) with daily financial data (flat average).
    This tests whether daily financial data adds value beyond macro indicators.
    """
    
    def __init__(self, h: int = 1, p_y: int = 1, m: int = 63):
        self.h = h
        self.p_y = p_y
        self.m = m
        self.params = None
        self.fitted = False
    
    def _flat_aggregate(self, block: np.ndarray) -> float:
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return np.nan
        return float(np.mean(block[valid]))
    
    def _extract_last_m(self, series: pd.Series, end_date: pd.Timestamp) -> np.ndarray:
        s = series.loc[:end_date].dropna()
        arr = np.full(self.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.m:].values
        arr[-len(take):] = take
        return arr
    
    def _build_rows(self, yq: pd.Series, fq: pd.Series, x_daily: pd.Series) -> Tuple[list, np.ndarray, list]:
        """Build training rows with factor and daily data."""
        yq = yq.dropna().sort_index()
        fq = fq.dropna().sort_index()
        
        common_idx = yq.index.intersection(fq.index)
        yq = yq.loc[common_idx]
        fq = fq.loc[common_idx]
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.p_y:
                continue
            
            y_lags = []
            for j in range(self.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.p_y:
                continue
            
            f_t = float(fq.iloc[i])
            if np.isnan(f_t):
                continue
            
            block = self._extract_last_m(x_daily, t)
            valid_count = (~np.isnan(block)).sum()
            if valid_count < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            rows.append((y_lags, f_t, x_agg))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, fq: pd.Series, x_daily: pd.Series) -> bool:
        """Fit FADL model using OLS."""
        rows, y, _ = self._build_rows(yq, fq, x_daily)
        
        if len(rows) < 10:
            return False
        
        n = len(rows)
        n_cols = 1 + self.p_y + 2  # const + AR lags + factor + x_agg
        X = np.zeros((n, n_cols))
        
        for i, (y_lags, f_t, x_agg) in enumerate(rows):
            X[i, 0] = 1.0
            X[i, 1:1+self.p_y] = y_lags
            X[i, 1+self.p_y] = f_t
            X[i, -1] = x_agg
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.params = {
                "const": beta[0],
                "rho": beta[1:1+self.p_y],
                "gamma": beta[1+self.p_y],
                "beta": beta[-1]
            }
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        fq: pd.Series,
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """Generate recursive OOS forecasts."""
        yq = yq.dropna().sort_index()
        fq = fq.dropna().sort_index()
        
        common_idx = yq.index.intersection(fq.index)
        yq_aligned = yq.loc[common_idx]
        fq_aligned = fq.loc[common_idx]
        q_dates = yq_aligned.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            y_train = yq_aligned.loc[:train_end]
            f_train = fq_aligned.loc[:train_end]
            rows, y_vec, _ = self._build_rows(y_train, f_train, x_daily)
            
            if len(rows) < min_train_obs:
                continue
            
            model = FADLModel(h=self.h, p_y=self.p_y, m=self.m)
            if not model.fit(y_train, f_train, x_daily):
                continue
            
            y_lags = [float(yq_aligned.iloc[i-j]) for j in range(self.p_y) if i-j >= 0]
            if len(y_lags) != self.p_y:
                continue
            
            f_t = float(fq_aligned.iloc[i])
            if np.isnan(f_t):
                continue
            
            block = self._extract_last_m(x_daily, train_end)
            if (~np.isnan(block)).sum() < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += model.params["gamma"] * f_t
            pred += model.params["beta"] * x_agg
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
                "gamma": model.params["gamma"],
                "beta": model.params["beta"],
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# DIEBOLD-MARIANO TEST (Andreou et al. 2013, Table 2)
# =============================================================================

class FARModelWithLeads:
    """
    Factor AR model with LEADS on monthly macro factor.
    
    Model: y_{t+h} = c + ρ·y_t + Σ_{j=0}^{J_M-1} γ_j·F_{m-j,t+1} + ε
    
    Where F_{m-j,t+1} is the j-th month (counting backward) of the macro
    indicator within quarter t+1 (the nowcast quarter).
    
    For FAR(J_CFNAI^M = 1): uses 1 month of leads (month 1 or 2 of target quarter)
    For FAR(J_NAPMNOI^M = 2): uses 2 months of leads
    
    From Table 3: FAR(J_CFNAI^M = 1), FAR(J_NAPMNOI^M = 2)
    """
    
    def __init__(self, h: int = 1, p_y: int = 1, lead_months: int = 1):
        """
        Initialize FAR model with leads.
        
        Parameters
        ----------
        h : int
            Forecast horizon
        p_y : int
            Number of AR lags
        lead_months : int
            Number of months of leads on macro indicator (1 or 2)
        """
        self.h = h
        self.p_y = p_y
        self.lead_months = lead_months
        self.params = None
        self.fitted = False
    
    def _get_macro_with_leads(
        self, 
        f_monthly: pd.Series, 
        target_quarter_end: pd.Timestamp
    ) -> np.ndarray:
        """
        Get monthly macro values with leads for the target quarter.
        
        Returns array of length lead_months with values from target quarter.
        """
        # Get first day of target quarter
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        
        leads = []
        for j in range(self.lead_months):
            # Month j of target quarter
            month_date = quarter_start + pd.DateOffset(months=j)
            month_end = month_date + pd.offsets.MonthEnd(0)
            
            # Get value available up to month_end
            s = f_monthly.loc[:month_end].dropna()
            if len(s) > 0:
                leads.append(float(s.iloc[-1]))
            else:
                leads.append(np.nan)
        
        return np.array(leads)
    
    def _build_rows(self, yq: pd.Series, f_monthly: pd.Series) -> Tuple[list, np.ndarray, list]:
        """Build training rows with monthly macro leads."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.p_y:
                continue
            
            # AR lags
            y_lags = []
            for j in range(self.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.p_y:
                continue
            
            # Monthly macro with leads
            f_leads = self._get_macro_with_leads(f_monthly, t_target)
            if np.any(np.isnan(f_leads)):
                continue
            
            rows.append((y_lags, f_leads))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, f_monthly: pd.Series) -> bool:
        """Fit FAR model with leads using OLS."""
        rows, y, _ = self._build_rows(yq, f_monthly)
        
        if len(rows) < 10:
            return False
        
        n = len(rows)
        # const + AR lags + lead_months gamma coefficients
        n_cols = 1 + self.p_y + self.lead_months
        X = np.zeros((n, n_cols))
        
        for i, (y_lags, f_leads) in enumerate(rows):
            X[i, 0] = 1.0
            X[i, 1:1+self.p_y] = y_lags
            X[i, 1+self.p_y:] = f_leads
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.params = {
                "const": beta[0],
                "rho": beta[1:1+self.p_y],
                "gamma": beta[1+self.p_y:]  # Array of gamma coefficients
            }
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        f_monthly: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """Generate recursive OOS forecasts."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            y_train = yq.loc[:train_end]
            rows, _, _ = self._build_rows(y_train, f_monthly)
            
            if len(rows) < min_train_obs:
                continue
            
            model = FARModelWithLeads(h=self.h, p_y=self.p_y, lead_months=self.lead_months)
            if not model.fit(y_train, f_monthly):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.p_y) if i-j >= 0]
            if len(y_lags) != self.p_y:
                continue
            
            f_leads = self._get_macro_with_leads(f_monthly, target_date)
            if np.any(np.isnan(f_leads)):
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += float(np.dot(model.params["gamma"], f_leads))
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


class FADLModelWithLeads:
    """
    Factor ADL model with LEADS on monthly macro factor (flat aggregation for daily).
    
    Model: y_{t+h} = c + ρ·y_t + Σ_{j} γ_j·F_{m-j,t+1} + β·x̄_t + ε
    
    From Table 3: FADL(J_CFNAI^M = 1), FADL(J_NAPMNOI^M = 2)
    
    This uses leads on monthly macro but NO leads on daily (x̄_t from quarter t).
    """
    
    def __init__(self, h: int = 1, p_y: int = 1, m: int = 63, lead_months_macro: int = 1):
        self.h = h
        self.p_y = p_y
        self.m = m
        self.lead_months_macro = lead_months_macro
        self.params = None
        self.fitted = False
    
    def _flat_aggregate(self, block: np.ndarray) -> float:
        valid = ~np.isnan(block)
        if valid.sum() == 0:
            return np.nan
        return float(np.mean(block[valid]))
    
    def _extract_last_m(self, series: pd.Series, end_date: pd.Timestamp) -> np.ndarray:
        s = series.loc[:end_date].dropna()
        arr = np.full(self.m, np.nan)
        if len(s) == 0:
            return arr
        take = s.iloc[-self.m:].values
        arr[-len(take):] = take
        return arr
    
    def _get_macro_with_leads(
        self, 
        f_monthly: pd.Series, 
        target_quarter_end: pd.Timestamp
    ) -> np.ndarray:
        """Get monthly macro values with leads."""
        quarter_start = target_quarter_end - pd.DateOffset(months=2)
        quarter_start = quarter_start.replace(day=1)
        
        leads = []
        for j in range(self.lead_months_macro):
            month_date = quarter_start + pd.DateOffset(months=j)
            month_end = month_date + pd.offsets.MonthEnd(0)
            
            s = f_monthly.loc[:month_end].dropna()
            if len(s) > 0:
                leads.append(float(s.iloc[-1]))
            else:
                leads.append(np.nan)
        
        return np.array(leads)
    
    def _build_rows(self, yq: pd.Series, f_monthly: pd.Series, x_daily: pd.Series) -> Tuple[list, np.ndarray, list]:
        """Build training rows."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        rows, y_out, target_dates = [], [], []
        
        for i, t in enumerate(q_dates):
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            t_target = q_dates[target_idx]
            
            if i < self.p_y:
                continue
            
            y_lags = []
            for j in range(self.p_y):
                lag_idx = i - j
                if lag_idx < 0:
                    break
                y_lags.append(float(yq.iloc[lag_idx]))
            
            if len(y_lags) != self.p_y:
                continue
            
            # Monthly macro with leads
            f_leads = self._get_macro_with_leads(f_monthly, t_target)
            if np.any(np.isnan(f_leads)):
                continue
            
            # Daily data (no leads - from quarter t)
            block = self._extract_last_m(x_daily, t)
            valid_count = (~np.isnan(block)).sum()
            if valid_count < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            rows.append((y_lags, f_leads, x_agg))
            y_out.append(float(yq.loc[t_target]))
            target_dates.append(t_target)
        
        return rows, np.array(y_out), target_dates
    
    def fit(self, yq: pd.Series, f_monthly: pd.Series, x_daily: pd.Series) -> bool:
        """Fit FADL model with macro leads using OLS."""
        rows, y, _ = self._build_rows(yq, f_monthly, x_daily)
        
        if len(rows) < 10:
            return False
        
        n = len(rows)
        n_cols = 1 + self.p_y + self.lead_months_macro + 1  # const + AR + gamma + beta
        X = np.zeros((n, n_cols))
        
        for i, (y_lags, f_leads, x_agg) in enumerate(rows):
            X[i, 0] = 1.0
            X[i, 1:1+self.p_y] = y_lags
            X[i, 1+self.p_y:1+self.p_y+self.lead_months_macro] = f_leads
            X[i, -1] = x_agg
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.params = {
                "const": beta[0],
                "rho": beta[1:1+self.p_y],
                "gamma": beta[1+self.p_y:1+self.p_y+self.lead_months_macro],
                "beta": beta[-1]
            }
            self.fitted = True
            return True
        except:
            return False
    
    def recursive_forecast(
        self, 
        yq: pd.Series, 
        f_monthly: pd.Series,
        x_daily: pd.Series, 
        start_date: pd.Timestamp,
        min_train_obs: int = 20
    ) -> pd.DataFrame:
        """Generate recursive OOS forecasts."""
        yq = yq.dropna().sort_index()
        q_dates = yq.index.tolist()
        
        out = []
        
        for i, train_end in enumerate(q_dates):
            if train_end < start_date:
                continue
            
            target_idx = i + self.h
            if target_idx >= len(q_dates):
                continue
            target_date = q_dates[target_idx]
            
            y_train = yq.loc[:train_end]
            rows, _, _ = self._build_rows(y_train, f_monthly, x_daily)
            
            if len(rows) < min_train_obs:
                continue
            
            model = FADLModelWithLeads(h=self.h, p_y=self.p_y, m=self.m, 
                                        lead_months_macro=self.lead_months_macro)
            if not model.fit(y_train, f_monthly, x_daily):
                continue
            
            # Forecast
            y_lags = [float(yq.iloc[i-j]) for j in range(self.p_y) if i-j >= 0]
            if len(y_lags) != self.p_y:
                continue
            
            f_leads = self._get_macro_with_leads(f_monthly, target_date)
            if np.any(np.isnan(f_leads)):
                continue
            
            block = self._extract_last_m(x_daily, train_end)
            if (~np.isnan(block)).sum() < self.m * 0.5:
                continue
            
            x_agg = self._flat_aggregate(block)
            if np.isnan(x_agg):
                continue
            
            pred = model.params["const"] + float(np.dot(model.params["rho"], np.array(y_lags)))
            pred += float(np.dot(model.params["gamma"], f_leads))
            pred += model.params["beta"] * x_agg
            
            out.append({
                "forecast_origin": train_end,
                "target_date": target_date,
                "y_true": float(yq.loc[target_date]),
                "y_pred": float(pred),
            })
        
        if len(out) == 0:
            return pd.DataFrame(columns=["forecast_origin", "target_date", "y_true", "y_pred", "fe", "se"])
        
        df = pd.DataFrame(out).sort_values("target_date")
        df["fe"] = df["y_true"] - df["y_pred"]
        df["se"] = df["fe"]**2
        return df


# =============================================================================
# AIC LAG SELECTION (Andreou et al. 2013)
# =============================================================================

def select_ar_lag_aic(yq: pd.Series, max_p: int = 4) -> int:
    """
    Select optimal AR lag order using AIC.
    
    AIC = n * log(RSS/n) + 2 * (p + 1)
    
    Parameters
    ----------
    yq : pd.Series
        Quarterly GDP series
    max_p : int
        Maximum lag order to consider
        
    Returns
    -------
    optimal_p : int
        Optimal lag order (minimizes AIC)
    """
    y = yq.dropna().values
    n = len(y)
    
    best_aic = np.inf
    best_p = 1
    
    for p in range(1, max_p + 1):
        if n <= p + 1:
            continue
        
        # Build regression: y_t = c + Σ rho_j * y_{t-j}
        Y = y[p:]
        X = np.column_stack([np.ones(len(Y))] + [y[p-j-1:-j-1 if j < p-1 else None] for j in range(p)])
        
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid = Y - X @ beta
            rss = np.sum(resid**2)
            
            # AIC
            n_obs = len(Y)
            k = p + 1  # Number of parameters (p AR + constant)
            aic = n_obs * np.log(rss / n_obs) + 2 * k
            
            if aic < best_aic:
                best_aic = aic
                best_p = p
        except:
            continue
    
    return best_p


# =============================================================================
# FORECAST COMBINATION FOR MULTIPLE ASSETS/FACTORS
# =============================================================================

def combine_all_forecasts_msfe(
    all_forecasts: Dict[str, pd.DataFrame],
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    delta: float = 0.9,
    kappa: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine forecasts from multiple assets/factors using discounted MSFE weights.
    
    This is for combining forecasts across 64 daily assets or 5 daily factors.
    Each asset/factor has its own forecast, and we combine them using
    MSFE-weighted averaging (Eq. 4.2-4.3 of paper).
    
    Parameters
    ----------
    all_forecasts : dict
        Dictionary {asset_name: DataFrame with y_true, y_pred columns}
    y_true_col, y_pred_col : str
        Column names
    delta : float
        Discount factor (0.9 in paper)
    kappa : float
        Exponent (2.0 in paper main spec)
        
    Returns
    -------
    combined_df : pd.DataFrame
        DataFrame with combined forecasts
    weights_df : pd.DataFrame
        Time-varying weights for each asset/factor
    """
    # Get common dates across all forecasts
    all_dates = None
    for name, fc in all_forecasts.items():
        fc = fc.set_index("target_date") if "target_date" in fc.columns else fc
        dates = set(fc.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    all_dates = sorted(list(all_dates))
    
    if len(all_dates) < 3:
        return pd.DataFrame(), pd.DataFrame()
    
    # Build aligned DataFrame
    aligned_data = {"y_true": []}
    for name in all_forecasts.keys():
        aligned_data[name] = []
    
    for date in all_dates:
        # Get y_true (should be same across all forecasts)
        first_fc = list(all_forecasts.values())[0]
        if "target_date" in first_fc.columns:
            first_fc = first_fc.set_index("target_date")
        y_true = first_fc.loc[date, y_true_col]
        aligned_data["y_true"].append(y_true)
        
        # Get predictions from each model
        for name, fc in all_forecasts.items():
            if "target_date" in fc.columns:
                fc = fc.set_index("target_date")
            try:
                aligned_data[name].append(fc.loc[date, y_pred_col])
            except:
                aligned_data[name].append(np.nan)
    
    df = pd.DataFrame(aligned_data, index=all_dates)
    
    # Use existing combine_forecasts_msfe
    model_cols = [c for c in df.columns if c != "y_true"]
    combined, weights, _ = combine_forecasts_msfe(
        df, delta=delta, kappa=kappa, h=1, T0=all_dates[0]
    )
    
    # Build result
    result_df = pd.DataFrame({
        "target_date": all_dates,
        "y_true": df["y_true"].values,
        "y_pred": combined.values
    })
    result_df["fe"] = result_df["y_true"] - result_df["y_pred"]
    result_df["se"] = result_df["fe"]**2
    
    return result_df, weights


def combine_forecasts_msfe(
    forecasts: pd.DataFrame, 
    delta: float = 0.9, 
    kappa: float = 2.0,
    h: int = 1,
    T0: pd.Timestamp = None,
    verbose: bool = False
) -> tuple:
    """
    Combine forecasts using discounted MSFE weights.
    
    Following Andreou, Ghysels & Kourtellos (2013), Equations 4.2-4.3:
    
    ω^{(h)}_{i,t} = (λ_{i,t}^{-1})^κ / Σ_j (λ_{j,t}^{-1})^κ
    
    where the discounted MSFE is:
    
    λ_{i,t} = Σ_{τ=T0}^{t-h} δ^{t-h-τ} (Y_{τ+h} - Ŷ_{i,τ+h|τ})²
    
    CRITICAL: The sum goes to t-h because at date t we don't observe Y_{t+h} yet.
    
    Parameters
    ----------
    forecasts : pd.DataFrame
        DataFrame with columns for each model's forecasts and 'y_true'
        Index should be the TARGET dates (τ+h)
    delta : float
        Discount factor (default: 0.9 as in paper)
    kappa : float
        Exponent on inverse MSFE (default: 2.0 as in paper's main specification)
    h : int
        Forecast horizon in quarters (default: 1)
    T0 : pd.Timestamp
        Start of OOS period (2001:Q1 for long, 2006:Q1 for short)
    verbose : bool
        If True, print debugging info for first few dates
        
    Returns
    -------
    tuple: (combined_forecasts: pd.Series, weights_df: pd.DataFrame, diagnostics: dict)
    """
    df = forecasts.sort_index()
    model_cols = [c for c in df.columns if c != "y_true"]
    
    # Set T0 if not provided
    if T0 is None:
        T0 = df.index[0]
    
    # Filter to only include dates >= T0
    df = df[df.index >= T0]
    
    # Results storage
    combined = []
    weights_history = []
    diagnostics = {"lambda": [], "last_tau_used": [], "n_errors_used": []}
    
    EPSILON = 1e-10  # Numerical stability
    
    for i, target_date in enumerate(df.index):
        """
        At date t, we want to combine forecasts for Y_{t+h}.
        We can only use errors from τ where we have observed Y_{τ+h}.
        Since we're at date t, the last τ where Y_{τ+h} is available is τ = t - h.
        
        In terms of target dates in our DataFrame:
        - Current target_date = t+h (what we're forecasting)
        - We can use errors for targets up to date t = target_date - h quarters
        - But in our DataFrame indexed by target dates, this means indices < i
        
        For h=1: at forecast date t for target t+1, we use errors up to target t
                 (i.e., all previous entries in the DataFrame)
        """
        
        # Minimum observations needed: at least 1 error to compute MSFE
        # For safety, require at least 2
        if i < 2:
            combined.append(np.nan)
            weights_history.append({c: np.nan for c in model_cols})
            continue
        
        # Historical data: all observations BEFORE current (targets τ+h where τ ≤ t-h)
        # This is exactly df.iloc[:i] since i corresponds to target t+h
        hist = df.iloc[:i].dropna(subset=["y_true"])
        
        if len(hist) < 1:
            combined.append(np.nan)
            weights_history.append({c: np.nan for c in model_cols})
            continue
        
        n_hist = len(hist)
        last_tau_plus_h = hist.index[-1]  # Last available target date
        
        # Discounted squared errors
        # Paper formula: δ^{t-h-τ} where τ goes from T0 to t-h
        # In our indexing: the most recent error (τ = t-h) gets weight δ^0 = 1
        #                  the oldest error gets weight δ^{n_hist-1}
        # So for index s in 0..n_hist-1: weight = δ^{n_hist - 1 - s}
        disc_weights = np.array([delta ** (n_hist - 1 - s) for s in range(n_hist)])
        
        y_true = hist["y_true"].values
        
        # Compute λ (discounted MSFE) for each model
        lambda_vals = {}
        for model in model_cols:
            preds = hist[model].values
            if np.any(np.isnan(preds)):
                continue
            squared_errors = (y_true - preds) ** 2
            lambda_i = float(np.sum(disc_weights * squared_errors))
            
            # Numerical stability: avoid λ = 0
            lambda_vals[model] = max(lambda_i, EPSILON)
        
        if len(lambda_vals) == 0:
            combined.append(np.nan)
            weights_history.append({c: np.nan for c in model_cols})
            continue
        
        # Compute weights: ω_i = (1/λ_i)^κ / Σ_j (1/λ_j)^κ
        raw_weights = {}
        for model, lam in lambda_vals.items():
            raw_weights[model] = (1.0 / lam) ** kappa
        
        total_weight = sum(raw_weights.values())
        
        # Normalize weights to sum to 1
        weights = {model: w / total_weight for model, w in raw_weights.items()}
        
        # Verify weights sum to 1
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-8, f"Weights don't sum to 1: {weight_sum}"
        
        # Combined forecast
        y_combined = sum(weights.get(c, 0.0) * df.loc[target_date, c] 
                         for c in model_cols if c in weights)
        
        combined.append(y_combined)
        weights_history.append(weights)
        
        # Store diagnostics
        diagnostics["lambda"].append(lambda_vals)
        diagnostics["last_tau_used"].append(last_tau_plus_h)
        diagnostics["n_errors_used"].append(n_hist)
        
        # Verbose output for debugging
        if verbose and i < 5:
            print(f"\n[DEBUG] Target: {target_date.date()}")
            print(f"  Last τ+h used: {last_tau_plus_h.date()} (n_errors={n_hist})")
            print(f"  λ values: {lambda_vals}")
            print(f"  Weights: {weights}")
            print(f"  Σω = {weight_sum:.6f}")
    
    # Build results
    combined_series = pd.Series(combined, index=df.index, name="y_pred_combined")
    weights_df = pd.DataFrame(weights_history, index=df.index)
    
    return combined_series, weights_df, diagnostics
