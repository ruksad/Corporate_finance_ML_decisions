# Aurora Finance — ML-Driven Corporate Finance Decisions

**Institution:** IIT Bombay — Executive PGD (Capstone Project)
**Company:** Aurora Finance (fictional)
**Date:** May 2026

---

## Project Overview

Aurora Finance is a fictional financial services firm that faces four core decision-making challenges across its business lines. This capstone applies end-to-end machine learning to each challenge — from raw data ingestion through model training, evaluation, and actionable business output.

The project is structured as four independent modules, each implemented in a self-contained Jupyter notebook.

| Module | Business Problem | ML Approach | Key Outcome |
|--------|-----------------|-------------|-------------|
| 1 | Which projects to fund? | XGBoost Regression + Classification + SHAP | 74 projects ranked; top capital allocations identified |
| 2 | Which loans to approve? Who is committing fraud? | PD Regression + Risk Scoring + Ensemble Fraud Detection | 74 approved, 26 rejected; 197 fraud alerts raised |
| 3 | Which stocks to invest in? | XGBoost + LightGBM + Random Forest Ensemble + Portfolio Optimisation | All 6 stocks SELL; cash recommended — validated by 35.9% alpha |
| 4 | How to price NIFTY options? | Gradient Boosting + Black-Scholes Hybrid | Call R² = 0.9984, Put R² = 0.9989 |

---

## Environment Setup

```bash
pyenv install 3.12.8
pyenv virtualenv 3.12.8 cfml-capstone
pyenv local cfml-capstone
pyenv activate cfml-capstone

pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn plotly scipy statsmodels imbalanced-learn ipykernel

# Register kernel for VS Code / Jupyter
python -m ipykernel install --user --name cfml-capstone --display-name "Python (cfml-capstone)"
```

Open notebooks in VS Code or Jupyter and select the **Python (cfml-capstone)** kernel.

To run a notebook non-interactively:
```bash
jupyter nbconvert --to notebook --execute StepN_<name>.ipynb --output StepN_<name>_out.ipynb
```

---

## Module 1 — Corporate Project Funding

**Notebook:** `Step1_corporateFinance_project_funding.ipynb`
**Data:** `sample_data/corporate_projects.csv` (50 projects)

### Problem
Aurora's investment committee must allocate limited capital across internal projects. The challenge is three-fold: forecast cash flows, predict project success probability, and rank projects for optimal capital deployment.

### Tasks
- **Task 1 — Cash Flow Forecasting (Regression):** Predict total 3-year cashflow per project using Ridge, Lasso, and XGBoost with LOOCV (chosen because n=50 is too small for k-fold). A naive mean-baseline ensures the model adds real signal.
- **Task 2 — Success Classification:** Predict probability of positive NPV using 5 domain-engineered features (NPV, IRR, PI, risk-adjusted ROI, market-adjusted ROI). Three classifiers compared via 5-fold stratified CV with AUC as primary metric.
- **Task 3 — Capital Allocation:** Rank all projects by expected value (predicted cashflow × success probability) and recommend top-priority investments with risk-tiered allocations.

### Key Design Choices
- Features capped at 4 for regression (samples-to-features ratio ≥ 10) to prevent overfitting on small data.
- XGBoost winner feeds SHAP explainability — each project gets a feature-attribution breakdown for the investment committee.
- Source cashflow columns dropped immediately after target construction to prevent label leakage.

### Outputs
`output/step1_predictions.csv`, SHAP summary plots, model evaluation charts.

---

## Module 2 — Banking: Loan Portfolio & Fraud Detection

**Notebook:** `Step2_banking_module_loan_portfolio_fraud_detect.ipynb`
**Data:** `sample_data/loan_portfolio.csv` (100 loans), `sample_data/transactions.csv` (200 transactions)

### Problem
Aurora's banking arm needs to (1) estimate default probability for each loan, (2) classify loan risk and detect fraudulent transactions, and (3) produce final lending decisions with interest rate adjustments.

### Three-Stage Pipeline (must run top-to-bottom)

**Stage 1 — PD Prediction (Regression):**
Ridge vs Lasso with LOOCV on 100 loans. Winner's predicted probability of default (PD) feeds into Stage 2 as an additional feature. LOOCV is used because n=100 is too small for k-fold.

**Stage 2 — Risk Scoring + Fraud Detection:**
- Weighted risk scoring engine converts raw loan attributes into a composite risk score.
- Logistic Regression vs Decision Tree (5-fold CV) classifies loans into Low / Medium / High risk.
- Separate fraud detection pipeline: Logistic Regression + Isolation Forest ensemble on transaction data, with F2-score threshold optimization (recall-weighted, since false negatives are costly in fraud).

**Stage 3 — Lending Decision Engine:**
Merges risk class and fraud signals to produce APPROVE / CONDITIONAL / REJECT decisions with corresponding interest rate adjustments.

### Outputs
`output/aurora_flagged_transactions.csv`, `sample_data/aurora_task3_lending_decisions.csv`, dashboard PNGs.

---

## Module 3 — Financial Markets: Portfolio Strategy

**Notebook:** `Step3_financial_markets_investment_strategies.ipynb`
**Data:** `sample_data/Market_Data_Revised.xlsx` — 6 Indian stocks (2005–2026), sheets `Comp 1`–`Comp 6` plus macro data.

### Problem
Aurora's asset management desk needs to decide which of 6 Indian equities to hold and in what weights, targeting superior risk-adjusted returns vs a cash (fixed deposit) benchmark.

### Architecture

**Ensemble Model:**
XGBoost + LightGBM + Random Forest with softmax-weighted blending based on each model's validation Sharpe ratio.

**Dual-Horizon Prediction:**
- 1-year forward return (`fwd_ret_1y`): rolling window CV, 14 folds, 5yr train / 1yr test.
- 3-year forward return (`fwd_ret_3y`): same structure with longer embargo.
- Embargo = horizon days between train and test to prevent label overlap leakage.

**Leakage Guards:**
All features use `.shift(1)` so row t only sees data available at t-1. Forward targets use `shift(-horizon)` so row t stores the return from t to t+horizon.

**Portfolio Optimisation:**
Max-Sharpe (SLSQP solver) with confidence-shrinkage applied to predicted returns. Final weights = 50/50 blend of 1Y and 3Y optimal portfolios.

**Backtest:**
2-year cumulative return vs cash (risk-free rate = 6.5% p.a.). BUY signal triggered only when predicted annualized return > RF_RATE. Result: all 6 stocks signalled SELL; the recommended cash position delivered 35.9% alpha over the backtest period.

### Key Constants
`RF_RATE = 0.065`, `HORIZON_1Y = 252`, `HORIZON_3Y = 756`

### Outputs
`output/backtest_portfolio.csv`, portfolio weight charts, return forecast plots.

---

## Module 4 — Derivatives: NIFTY Options Pricing

**Notebook:** `Step4_derivative_module.ipynb`
**Data:** `sample_data/Option_Chain_NSE_intraday_NIFTY_2Feb24.xlsx` — NIFTY 50 intraday option chain, 2 Feb 2024.

### Problem
Aurora's derivatives desk needs to price NIFTY options accurately, understand where Black-Scholes misprices, and construct hedged portfolios using ML-corrected valuations.

### Four Sections

**Section 1 — ML Pricing:**
Gradient Boosting Regressor (separate models for calls and puts) trained on intraday option chain data with an 80/20 time-based split. Achieves Call R² = 0.9984, Put R² = 0.9989.

**Section 2 — Black-Scholes Pricing:**
Classic BS pricing using market implied volatility (IV). Error analysis compares BS price vs actual LTP to identify systematic mispricing patterns.

**Section 3 — Hedging Strategies:**
- ATM straddle construction.
- Bull call spread.
- Delta-neutral portfolio using ML-predicted Greeks (delta, gamma).

**Section 4 — Risk Dashboard:**
Side-by-side ML vs BS accuracy metrics, IV skew visualised by moneyness bucket, and BS mispricing plotted by time of day.

### Key Constants
`R = 0.065` (risk-free rate), `LOT = 50` (NIFTY lot size)

---

## Repository Structure

```
.
├── Step1_corporateFinance_project_funding.ipynb
├── Step2_banking_module_loan_portfolio_fraud_detect.ipynb
├── Step3_financial_markets_investment_strategies.ipynb
├── Step4_derivative_module.ipynb
├── sample_data/
│   ├── corporate_projects.csv
│   ├── loan_portfolio.csv
│   ├── transactions.csv
│   ├── Market_Data_Revised.xlsx
│   └── Option_Chain_NSE_intraday_NIFTY_2Feb24.xlsx
├── output/
│   ├── step1_predictions.csv
│   ├── aurora_flagged_transactions.csv
│   └── backtest_portfolio.csv
└── Aurora_Finance_Project_Report.md
```

**Important:** Notebooks are independent — each reloads raw data from scratch. No shared state exists between modules. Within Module 2, cells must be run top-to-bottom as Task 3 depends on objects built by Tasks 1 and 2.
