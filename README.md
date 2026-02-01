# GDP Forecasting with MIDAS Regression

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Econometrics-MIDAS-green.svg" alt="MIDAS">
  <img src="https://img.shields.io/badge/Paris%20Dauphine-PSL-red.svg" alt="Dauphine">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
</p>

**Master 2 272 Quantitative Management - Université Paris Dauphine-PSL**

> Academic replication and extension of Andreou, Ghysels, Kourtellos (2013):  
> *"Should Macroeconomic Forecasters Use Daily Financial Data and How?"*  
> Journal of Business & Economic Statistics, 31(2), 240-251.

---

## 👥 Authors

| Name | Role |
|------|------|
| **Théo Verdelhan** | Lead Developer & Econometric Implementation |
| **Léo Renault** | Data Engineering & PCA Analysis |
| **Arthur Le Net** | Model Validation & Benchmarking |
| **Nicolas Annon** | Extension Development & Visualization |

---

## 📋 Project Overview

### Objective

This project implements and extends the **ADL-MIDAS (Autoregressive Distributed Lag - Mixed Data Sampling)** framework for forecasting **quarterly US GDP growth** using **high-frequency daily financial data**. The MIDAS approach addresses a fundamental challenge in macroeconomic forecasting: how to efficiently exploit the information content of variables sampled at higher frequencies than the target variable.

### Academic Context

This project was developed as part of the **Quantitative Management** course at **Paris Dauphine-PSL University**. The assignment required:

1. **Full replication** of the original research paper using our own data sources
2. **Critical analysis** of the methodology and results
3. **Extension** of the paper with novel contributions

---

## 🔬 Methodology

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

## 📊 Data

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

## 🏗️ Project Structure

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

## 🔄 Replication Results

### Table 1: RMSFE Comparisons (No Leads, h=1)

| Model | Long Sample | Short Sample |
|-------|-------------|--------------|
| **Random Walk** | 0.62 (baseline) | 0.45 (baseline) |
| **AR(1)** | 0.94 | 0.97 |
| **ADL-MIDAS (5 Daily Factors)** | 0.88 | 0.91 |
| **FADL-MIDAS (CFNAI + Daily Factors)** | 0.82 | 0.85 |
| **Forecast Combination** | 0.79 | 0.83 |

*Values < 1 indicate improvement over Random Walk*

### Key Findings from Replication

1. **MIDAS outperforms AR benchmarks** by 6-15% in RMSFE
2. **Daily financial factors** contain significant information for GDP forecasting
3. **Forecast combination** via MSFE weights provides additional gains
4. **Leads (nowcasting)** substantially improve accuracy when available
5. **PCA factor extraction** efficiently summarizes high-dimensional financial data

---

## 🚀 Extension: Post-Pandemic Period (2020-2025)

### Motivation

The original paper covers 1986-2008, ending before the Global Financial Crisis aftermath. We extend the analysis to the **2020-2025 period** to test:

1. Model robustness during **COVID-19 shock and recovery**
2. Predictive ability during **high inflation regime**
3. Performance under **aggressive Fed tightening**

### Extension Results

| Period | AR(1) RMSFE | MIDAS RMSFE | Improvement |
|--------|-------------|-------------|-------------|
| **Full 2024-2025** | 0.52 | 0.44 | +15.4% |
| **Post-COVID Recovery** | 0.48 | 0.41 | +14.6% |
| **High Inflation (2022-2023)** | 0.61 | 0.55 | +9.8% |

### Novel Contributions

1. **State-Dependent θ**: Time-varying MIDAS weights based on volatility regimes
2. **Multi-Horizon Analysis**: h = 1, 2, 3, 4 quarters ahead
3. **Real-Time Nowcasting**: Using daily leads for current quarter estimation
4. **Extended Macro Indicators**: ADS, CFNAI, PMI integration

---

## 📈 Visualizations

The notebook generates comprehensive visualizations saved to `PLOT ANALYSIS/`:

- **Data Exploration**: Asset class coverage, stationarity tests, volatility analysis
- **Factor Analysis**: PCA loadings, variance explained, GDP correlations
- **Forecast Comparison**: MIDAS vs AR(1) vs Random Walk time series
- **MIDAS Weights**: Exponential Almon weight decay patterns
- **Sub-Period Performance**: Crisis vs normal periods breakdown
- **Recent Period**: 2024-2025 nowcasting results

---

## 🛠️ Installation

### Option 1: Automatic Setup (Recommended)

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
.\setup.bat
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

# Activate (Windows)
.venv\Scripts\activate

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

## 🚀 Usage

1. **Activate environment:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Run the notebook:**
   - VS Code: Open `main.ipynb`, select `.venv` as kernel
   - Jupyter: `jupyter notebook main.ipynb`

3. **Execute all cells** — outputs are saved to `PLOT ANALYSIS/`

---

## 📚 References

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
