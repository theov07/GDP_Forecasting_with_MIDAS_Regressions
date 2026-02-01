# GDP Forecasting with MIDAS Regression

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Econometrics-MIDAS-green.svg" alt="MIDAS">
  <img src="https://img.shields.io/badge/Paris%20Dauphine-PSL-red.svg" alt="Dauphine">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
</p>

**Master 2 272 Quantitative Finance - Université Paris Dauphine-PSL**

> Academic replication and extension of Andreou, Ghysels, Kourtellos (2013):  
> *"Should Macroeconomic Forecasters Use Daily Financial Data and How?"*  
> Journal of Business & Economic Statistics, 31(2), 240-251.

---

## Authors

| Name |
|------|
| **Théo Verdelhan** | **Léo Renault** | **Arthur Le Net** | **Nicolas Annon** |

---

## Project Overview

### Objective

This project implements and extends the **ADL-MIDAS (Autoregressive Distributed Lag - Mixed Data Sampling)** framework for forecasting **quarterly US GDP growth** using **high-frequency daily financial data**. The MIDAS approach addresses a fundamental challenge in macroeconomic forecasting: how to efficiently exploit the information content of variables sampled at higher frequencies than the target variable.

### Academic Context

This project was developed as part of the **Quantitative Management** course at **Paris Dauphine-PSL University**. The assignment required:

1. **Full replication** of the original research paper using our own data sources
2. **Critical analysis** of the methodology and results
3. **Extension** of the paper with novel contributions

---

## Methodology

### The MIDAS Model

The core ADL-MIDAS specification estimates quarterly GDP growth as:

$$y_{t+h}^Q = \alpha + \sum_{k=1}^{p} \rho_k \cdot y_{t-k+1}^Q + \beta \cdot \sum_{j=0}^{m-1} B(j;\theta) \cdot x_{t-j}^D + \varepsilon_t$$

Where:
- $y^Q$: Quarterly GDP growth rate
- $x^D$: Daily financial variables (or PCA factors)
- $B(j;\theta) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{k} \exp(\theta_1 k + \theta_2 k^2)}$: Exponential Almon polynomial weights
- $h$: Forecast horizon (quarters ahead)
- $m$: Number of daily observations used (typically 63 ≈ 1 quarter)
- $p$: Autoregressive lag order (selected via AIC)

### Key Features of Our Implementation

1. **Exponential Almon Weighting**: Parsimonious aggregation of daily data with only 1-2 parameters
2. **Recursive Out-of-Sample Forecasting**: Expanding window estimation to avoid look-ahead bias
3. **AIC-Based Lag Selection**: Automatic selection of AR lag order
4. **MSFE-Weighted Forecast Combination**: Combining multiple predictors using discounted forecast errors

---

## Data

### Sources

We use **Bloomberg Terminal** data covering multiple asset classes:

| Asset Class | Examples | Frequency |
|------------|----------|-----------|
| **Equity Indices** | S&P 500, NASDAQ, Russell 2000 | Daily |
| **Fixed Income** | 2Y/10Y Treasury, Credit Spreads | Daily |
| **Commodities** | WTI Crude Oil, Gold, Copper | Daily |
| **Foreign Exchange** | EUR/USD, USD/JPY, DXY | Daily |
| **Macro Indicators** | CFNAI, ADS, ISM PMI | Monthly/Daily |
| **GDP Growth** | US Real GDP QoQ | Quarterly |

### Sample Periods (Following the Paper)

| Sample | Data Period | Training | Out-of-Sample |
|--------|-------------|----------|---------------|
| **Long Sample** | 1986-01-01 → 2008-12-31 | 1986Q1-2000Q4 | 2001Q1-2008Q4 |
| **Short Sample** | 1999-01-01 → 2008-12-31 | 1999Q1-2005Q4 | 2006Q1-2008Q4 |
| **Extended (Our Extension)** | 2020-01-01 → 2025-12-31 | 2020Q1-2023Q4 | 2024Q1-2025Q4 |

---

## Project Structure

```
QUANTITATIVE_MIDAS_MODEL_REPLICATION/
├── main.ipynb                 # Main analysis notebook (run this)
├── requirements.txt           # Python dependencies
├── setup.sh / setup.bat       # Environment setup scripts
├── README.md
│
├── DATAS/
│   └── bloomberg_all_tickers.csv   # Bloomberg financial data
│
├── FUNCTIONS/
│   ├── __init__.py
│   ├── data_loader.py         # Bloomberg CSV parsing with date handling
│   ├── transformations.py     # Log-returns, first differences, PCA
│   ├── midas.py               # MidasSpec, MidasModel, MidasModelWithLeads
│   ├── benchmarks.py          # AR(1), Random Walk, MSFE combination
│   ├── evaluation.py          # RMSFE computation, sub-period analysis
│   ├── analysis.py            # Sample period definitions, horizon tests
│   ├── table_replication_v2.py # Paper Table 1-5 replication
│   ├── plots.py               # Visualization functions
│   └── data_diagnostics.py    # Coverage checks, stationarity tests
│
├── PLOT ANALYSIS/             # Generated figures and tables
│
└── TOOLS/
    ├── install_requirement.py # Dependency installer
    └── convert_excel_to_csv.ipynb
```

---

## Replication Results

### Table 1: RMSFE Comparisons — No Leads (Ratios to RW)

| Model | Long h=1 | Long h=4 | Short h=1 | Short h=4 |
|-------|----------|----------|-----------|----------|
| **RW (absolute RMSFE)** | 2.69 | 3.18 | 3.46 | 4.66 |
| **AR** | 1.01 | 0.91 | 1.13 | 1.00 |
| **FAR (CFNAI)** | 0.91 | 0.90 | 0.94 | 0.98 |
| **ADL (5 DF)** | 1.09 | 1.12 | 1.20 | 1.14 |
| **ADL-MIDAS (5 DF)** | 1.11 | 1.11 | 1.24 | 1.13 |
| **FADL (CFNAI, 5 DF)** | 0.96 | 1.12 | 1.00 | 1.14 |
| **FADL-MIDAS (CFNAI, 5 DF)** | 1.07 | 0.86 | 1.02 | 1.00 |

*Values < 1 indicate improvement over Random Walk*

### Table 3: RMSFE Comparisons — With Leads (Ratios to RW)

| Model | Long h=1 | Long h=4 | Short h=1 | Short h=4 |
|-------|----------|----------|-----------|----------|
| **ADL-MIDAS (J_D=2)** | 0.97 | 0.87 | 0.94 | 0.89 |
| **FADL-MIDAS (J_D=2)** | 0.77 | 0.73 | 0.70 | 0.62 |
| **FADL-MIDAS (J_M=1, J_D=2)** | 0.93 | 0.81 | 0.86 | 0.82 |
| **FAR (J_M=1)** | 0.87 | 0.73 | 0.84 | 0.72 |
| **FADL (J_M=1)** | 0.90 | 0.88 | 0.92 | 0.86 |

### Table 5: ADS Index Comparisons (Ratios to RW)

| Model | Long h=1 | Long h=4 | Short h=1 | Short h=4 |
|-------|----------|----------|-----------|----------|
| **ADL-MIDAS (J_D,ADS=2)** | 0.57 | 0.48 | 0.56 | 0.42 |
| **FADL-MIDAS (J_M=1, J_D,ADS=2)** | 0.60 | 0.52 | 0.60 | 0.46 |

### Key Findings from Replication

1. **Leads are crucial**: Models with daily leads (J_D=2) substantially outperform no-lead specifications
2. **ADS dominates**: The Aruoba-Diebold-Scotti daily macro index achieves the best performance (40-50% improvement vs RW)
3. **CFNAI adds value**: Factor AR with CFNAI beats pure AR benchmark
4. **FADL-MIDAS with leads** is the best financial-factor model (0.70-0.77 vs RW)
5. **Short sample challenges**: Pure financial factors underperform on shorter estimation windows without leads

---

## Extension: Two-β MIDAS Model (Lags vs Leads)

### Motivation

In the standard MIDAS specification (Andreou et al., 2013), a **single β parameter** determines the weighting applied to the entire daily block—both lagged data and nowcast leads. This imposes the same weighting dynamics on historical information and intra-quarter data.

**Our contribution**: We propose a **Two-β MIDAS** extension that **separates** this parameter into:
- **β_lag**: Weights applied to the lag block (past data, m = 63 days ≈ 1 quarter)
- **β_lead**: Weights applied to the lead block (nowcast data, m_L ≈ 42 days ≈ 2 months)

This relaxes a potentially restrictive constraint while remaining parsimonious (only one additional parameter).

### Model Specification

The extended model at horizon h = 1 is:

$$y_{t+1} = \alpha + \rho y_t + \beta_{lag} \sum_{k=1}^{m} B(k;\theta_{lag}) x_{t-k} + \beta_{lead} \sum_{j=1}^{m_L} B(j;\theta_{lead}) x_{t+j} + \varepsilon_{t+1}$$

Where $B(k;\theta) = \frac{\exp(\theta k)}{\sum_\ell \exp(\theta \ell)}$ is the normalized exponential Almon weighting function.

### Economic Intuition

The leads correspond to the **beginning of the target quarter**—the most recent data available at forecast time. Under a single-β constraint, this valuable nowcast information may be under-weighted if the weighting shape is primarily driven by the lag block. Allowing **β_lag ≠ β_lead** lets the model treat historical and nowcast information differently, which is more economically coherent for nowcasting exercises.

### Results

#### Comparison: Two-β vs Single-β (OOS 2024-2025)

| Model | RMSFE | Rel. to RW | vs RW |
|-------|-------|------------|-------|
| **Two-β MIDAS (J_D = 2)** | 2.727 | 1.288 | -28.8% |
| Single-β MIDAS (J_D = 2) | 3.240 | 1.530 | -53.0% |

**→ Two-β improves upon Single-β by 15.8%**

#### Application to Paper Samples (Long & Short)

| Sample | Model | RMSFE | Rel. to RW | Improvement vs Single-β |
|--------|-------|-------|------------|------------------------|
| **Long** | Two-β MIDAS | 2.397 | 1.132 | **+26.0%** |
| **Short** | Two-β MIDAS | 2.827 | 1.335 | **+12.8%** |

#### Estimated Parameters (Average across OOS forecasts)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| θ_lag | 0.0266 | Decay rate for historical weights |
| θ_lead | 0.0176 | Decay rate for nowcast weights |
| β_lag | 2.5252 | Scale for lag contribution |
| β_lead | 3.6708 | Scale for lead contribution |

The distinct estimated dynamics confirm the value of separating lag and lead effects.

### Complete Model Ranking (OOS 2024-2025)

| Rank | Model | RMSFE | vs RW |
|------|-------|-------|-------|
| 1 | ADL(flat) | 0.951 | +55.1% |
| 2 | FAR(CFNAI) | 1.802 | +14.9% |
| 3 | AR | 1.884 | +11.0% |
| 4 | FADL(J_M = 1) | 2.114 | +0.2% |
| 5 | RW (baseline) | 2.117 | — |
| 6 | FAR(J_M = 1) | 2.261 | -6.8% |
| **7** | **⭐ Two-β MIDAS (J_D = 2)** | **2.727** | **-28.8%** |
| 8 | ADL-MIDAS(J_D = 2) | 3.240 | -53.0% |
| 9 | FADL-MIDAS(J_M=1, J_D=2) | 3.653 | -72.5% |
| 10 | FADL-MIDAS | 3.732 | -76.3% |

*⭐ indicates our novel contribution*

### Discussion

The Two-β extension **consistently improves** over the standard single-β MIDAS across all samples. The separate weighting profiles allow the model to:
1. Apply a **regular decay** on the lag block (classical fading memory)
2. Treat **nowcast information** with its own dynamics

**Limitations**: The extension does not beat RW on the recent 2024-2025 period. This is likely due to:
- Short OOS window (8 quarters), sensitive to individual large errors
- Post-COVID regime instability in the factor-GDP relationship

---

## Visualizations

The notebook generates comprehensive visualizations saved to `PLOT ANALYSIS/`:

- **Data Exploration**: Asset class coverage, stationarity tests, volatility analysis
- **Factor Analysis**: PCA loadings, variance explained, GDP correlations
- **Forecast Comparison**: MIDAS vs AR(1) vs Random Walk time series
- **MIDAS Weights**: Exponential Almon weight decay patterns
- **Sub-Period Performance**: Crisis vs normal periods breakdown
- **Recent Period**: 2024-2025 nowcasting results

---

## Installation

### Option 1: Automatic Setup (Recommended)

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```


### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/MIDAS-GDP-Forecasting.git
cd MIDAS-GDP-Forecasting

# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy`, `pandas`: Data manipulation
- `scipy`, `statsmodels`: Econometric estimation
- `scikit-learn`: PCA, preprocessing
- `matplotlib`: Visualization
- `jupyter`: Notebook execution

---

## Usage

1. **Activate environment:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Run the notebook:**
   - VS Code: Open `main.ipynb`, select `.venv` as kernel
   - Jupyter: `jupyter notebook main.ipynb`

3. **Execute all cells** — outputs are saved to `PLOT ANALYSIS/`

---

## References

### Primary Reference

Andreou, E., Ghysels, E., & Kourtellos, A. (2013). *Should Macroeconomic Forecasters Use Daily Financial Data and How?* Journal of Business & Economic Statistics, 31(2), 240-251. [DOI: 10.1080/07350015.2013.767199](https://doi.org/10.1080/07350015.2013.767199)

### Additional References

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). *The MIDAS Touch: Mixed Data Sampling Regression Models*
- Ghysels, E., Sinko, A., & Valkanov, R. (2007). *MIDAS Regressions: Further Results and New Directions*
- Stock, J.H., & Watson, M.W. (2002). *Macroeconomic Forecasting Using Diffusion Indexes*

---

## 🎓 Academic Information

**Course**: M2 272 - Quantitative Finance and Financial Engineering  
**Institution**: Université Paris Dauphine-PSL  
**Academic Year**: 2025-2026  

---

## 📄 License

This project is for academic purposes. The code is provided as-is for educational use.

---

## 📧 Contact

For questions about this project:
- **Théo Verdelhan** - theo.verdelhan@dauphine.eu

---

<p align="center">
  <i>Developed at Université Paris Dauphine-PSL</i>
</p>
