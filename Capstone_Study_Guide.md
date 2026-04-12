# Aurora Finance Capstone — Complete Study Guide & Workflow

## Project Overview

You are acting as a **Data & ML Strategy Consultant** for Aurora Finance, a mid-size financial services firm. The project spans 4 modules that mirror real finance divisions: Corporate Finance, Banking, Financial Markets, and Derivatives. Each module asks you to apply ML to make better business decisions — not just build models, but **translate model outputs into actionable executive recommendations**.

The evaluation weights tell you everything about what matters:

| Criteria | Weight | What it really means |
|---|---|---|
| ML Model Accuracy & Insights | 25% | Models must work, but insights matter more than raw accuracy |
| Business Decision Alignment | 25% | Can you convert predictions into "fund this / reject that" decisions? |
| Integration Across Modules | 20% | The 4 steps should connect into a coherent narrative |
| Explainability | 15% | SHAP plots, feature importance — executives must trust the model |
| Executive Communication | 15% | Clean visuals, concise summaries, no jargon |

---

# STEP 1: Corporate Finance Module — Project Funding

## The Business Problem

Aurora has a pool of internal projects competing for limited capital. Each project requires an upfront investment and promises uncertain future cash flows. Your job: **decide which projects to fund**.

This is the classic **capital budgeting** problem that every CFO faces.

## Your Data: `corporate_projects.csv`

50 projects with: Investment_Cost, Expected_Cashflow_Year1–3, Historical_ROI, Market_Growth, Project_Risk (Low/Medium/High), Department, and a **Success** label (1 = project was successful, 0 = it failed).

---

## Key Finance Concepts You Must Know

### 1. Net Present Value (NPV)

NPV is the single most important concept in corporate finance. The core idea: **a dollar today is worth more than a dollar tomorrow**, because you could invest that dollar and earn a return.

**The formula:**

```
NPV = -Investment_Cost + CF1/(1+r)^1 + CF2/(1+r)^2 + CF3/(1+r)^3
```

Where `r` is the **discount rate** (also called the cost of capital or hurdle rate). This represents the minimum return Aurora expects from any investment.

**Why it matters:** If NPV > 0, the project creates value — it returns more than what Aurora could earn by simply investing the money elsewhere at rate `r`. If NPV < 0, the money is better spent elsewhere.

**Choosing the discount rate:** In practice, firms use their Weighted Average Cost of Capital (WACC). For this project, you can assume a rate like 10% (common for mid-size firms) or experiment with 8–12% to see how sensitive your rankings are.

**Example from your data — Project 1:**
- Investment: ₹184,654
- Cash flows: ₹583,556 (Y1), ₹2,348,816 (Y2), ₹1,888,756 (Y3)
- At r = 10%: NPV = -184,654 + 583,556/1.1 + 2,348,816/1.21 + 1,888,756/1.331
- NPV = -184,654 + 530,505 + 1,941,170 + 1,418,975 = **₹3,705,996**
- This project has a massive positive NPV — but it was labeled Success=0! This is interesting — perhaps the cash flows didn't materialize as expected.

### 2. Internal Rate of Return (IRR)

IRR is the discount rate that makes NPV exactly zero. Think of it as the project's "built-in" rate of return.

```
0 = -Investment + CF1/(1+IRR)^1 + CF2/(1+IRR)^2 + CF3/(1+IRR)^3
```

You can't solve this algebraically — you need numerical methods (Python's `numpy.irr` or `scipy.optimize`).

**Decision rule:** If IRR > cost of capital (say 10%), the project is worth funding. The higher the IRR, the better.

**IRR vs NPV:** NPV tells you how much value a project creates in absolute terms. IRR tells you the percentage return. A small project with 50% IRR might create less total value than a large project with 15% IRR. For ranking, NPV is generally preferred; IRR is a useful complement.

### 3. Risk-Adjusted Return

Not all NPVs are equal. A project with NPV = ₹1M and High risk is less attractive than NPV = ₹1M and Low risk.

**Approaches:**
- **Risk-adjusted discount rate:** Use a higher `r` for risky projects (e.g., Low=8%, Medium=10%, High=14%). This naturally penalizes risky projects in NPV.
- **Expected Value (EV):** EV = P(Success) × NPV_if_success + P(Failure) × NPV_if_failure. Your ML model's predicted probability of success directly feeds into this.
- **Sharpe-like ratio:** NPV / Risk_score — simple but effective for ranking.

### 4. Profitability Index (PI)

```
PI = (NPV + Investment_Cost) / Investment_Cost = PV of Cash Flows / Investment_Cost
```

PI tells you how much value you get per rupee invested. When capital is constrained (you can't fund everything), PI helps you pick the combination of projects that maximizes total value.

---

## ML Tasks for Step 1

### Task 1: Regression — Forecast Cash Flows or NPV

**What to predict:** You can either predict individual year cash flows, or compute NPV first and predict that directly. I'd recommend computing NPV as a feature and then predicting Success (Task 2) — it's more useful for decision-making.

**Feature engineering ideas:**
- Compute NPV at different discount rates (8%, 10%, 12%)
- Compute IRR for each project
- Compute PI (Profitability Index)
- Compute total cash flow / investment ratio
- One-hot encode Department and Project_Risk
- Create interaction features: Market_Growth × Historical_ROI

**Models to use:**
- **Random Forest Regressor** — robust, handles non-linear relationships, gives feature importance for free
- **XGBoost Regressor** — often better accuracy, handles missing data gracefully
- **Linear Regression** — as a baseline; interpretable but may underfit

**Code skeleton:**
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('corporate_projects.csv')

# Compute NPV at 10% discount rate
r = 0.10
df['NPV'] = (-df['Investment_Cost'] 
             + df['Expected_Cashflow_Year1']/(1+r) 
             + df['Expected_Cashflow_Year2']/(1+r)**2 
             + df['Expected_Cashflow_Year3']/(1+r)**3)

df['Total_CF'] = df[['Expected_Cashflow_Year1','Expected_Cashflow_Year2','Expected_Cashflow_Year3']].sum(axis=1)
df['CF_to_Investment'] = df['Total_CF'] / df['Investment_Cost']
df['PI'] = (df['NPV'] + df['Investment_Cost']) / df['Investment_Cost']
```

### Task 2: Classification — Predict Project Success

This is the core ML task. The **Success** column (0/1) is your target.

**Why this matters for business:** If you can predict which projects will succeed *before* investing, you avoid burning capital on doomed projects.

**Models to use:**
- **Logistic Regression** — baseline, highly interpretable, gives probability directly
- **Random Forest Classifier** — good balance of accuracy and interpretability
- **XGBoost Classifier** — usually best accuracy on tabular data

**Important considerations:**
- With only 50 samples, **cross-validation is critical** — use 5-fold or leave-one-out
- Check class balance: how many 1s vs 0s? If imbalanced, use `class_weight='balanced'`
- Don't just report accuracy — use **precision, recall, F1, and AUC-ROC**
- The predicted **probability** (not just 0/1) is what feeds into your ranking

**Code skeleton:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

features = ['Investment_Cost', 'Expected_Cashflow_Year1', 'Expected_Cashflow_Year2',
            'Expected_Cashflow_Year3', 'Historical_ROI', 'Market_Growth', 'NPV', 'PI']
# One-hot encode categoricals
df_model = pd.get_dummies(df, columns=['Department', 'Project_Risk'])

X = df_model.drop(['Project_ID', 'Success'], axis=1)
y = df_model['Success']

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
```

### Task 3: Ranking — Recommend Top Projects

This is where finance meets ML. Combine your financial metrics with ML predictions:

```
Ranking Score = Predicted_P(Success) × NPV × Risk_Adjustment_Factor
```

Or use Expected Value:
```
EV = P(Success) × NPV_if_success + (1 - P(Success)) × (-Investment_Cost)
```

Present a ranked table showing: Project_ID, Department, NPV, IRR, P(Success), EV, Risk, and your Recommendation (Fund/Reject/Review).

### SHAP for Explainability

SHAP (SHapley Additive exPlanations) is essential. It answers: **why did the model predict this project would succeed?**

```python
import shap

model.fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot: which features matter most overall
shap.summary_plot(shap_values[1], X)  # [1] for the "Success" class

# Force plot: explain a single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0])
```

SHAP tells you things like: "Project 7 was predicted to succeed primarily because it has high Historical_ROI and positive Market_Growth, despite its Medium risk rating."

---

# STEP 2: Banking Module — Credit Risk & Fraud Detection

## The Business Problem

Aurora's banking division needs to: (a) assess which loans are likely to default, so it can price them correctly and set aside reserves, and (b) detect fraudulent transactions before they cause losses.

## Your Data

**`loan_portfolio.csv`** — 100 loans with borrower features and a continuous **PD** (Probability of Default) target.

**`transactions.csv`** — 200 transactions with a binary **Fraud_Flag** target.

---

## Key Finance/Banking Concepts

### 1. Probability of Default (PD)

PD is the likelihood that a borrower will fail to repay their loan within a given time horizon (usually 1 year). It's one of the three pillars of credit risk under the Basel framework:

- **PD** — Probability of Default (will they default?)
- **LGD** — Loss Given Default (if they default, how much do you lose? Typically 40-60%)
- **EAD** — Exposure at Default (how much is outstanding when they default?)

**Expected Loss = PD × LGD × EAD**

In your data, PD ranges from 0.01 to 0.30 (1% to 30%). This is a **regression** problem — you're predicting a continuous probability, not a binary outcome.

### 2. Credit Scoring and Risk Buckets

After predicting PD, you categorize loans into risk buckets:

| PD Range | Risk Rating | Action |
|---|---|---|
| 0–5% | Very Low Risk | Auto-approve, lowest interest rate |
| 5–10% | Low Risk | Approve with standard terms |
| 10–15% | Moderate Risk | Approve with higher rate or collateral |
| 15–20% | Elevated Risk | Manual review required |
| 20–25% | High Risk | Likely reject or require guarantor |
| 25%+ | Very High Risk | Reject |

These thresholds are yours to define — justify them based on your analysis.

### 3. Key Risk Factors in Your Data

- **Debt_to_Income (DTI):** The percentage of income going to debt payments. DTI > 0.4 is generally considered risky. Your data has values up to 0.90 — these are extremely leveraged borrowers.
- **Credit_History_Length:** Longer history = more data to assess = lower uncertainty. Short histories (1-2 years) are riskier.
- **Past_Default:** Binary flag. A previous default is one of the strongest predictors of future default.
- **Loan_Amount relative to Income:** A ₹900K loan on ₹500K income is very different from the same loan on ₹5M income.
- **Interest_Rate:** Higher rates are often charged to riskier borrowers (endogeneity alert — this can be both a cause and an effect).
- **Loan_Term_Months:** Longer terms = more time for things to go wrong.

### 4. Fraud Detection Concepts

Fraud detection is fundamentally different from credit risk:

**Class imbalance:** In your data, only ~8 of 200 transactions are fraudulent (4%). This is actually a mild imbalance — real-world fraud rates are often <0.1%. Still, accuracy is misleading (96% accuracy by predicting "no fraud" every time).

**Approaches:**
- **Supervised (since you have labels):** Train on Fraud_Flag using the transaction features
- **Unsupervised (anomaly detection):** Ignore labels and detect "unusual" transactions — useful when labels are incomplete

**Key anomaly indicators to engineer:**
- Transaction amount relative to customer's average
- Transaction frequency (multiple transactions in short time)
- Unusual transaction type for that customer
- Time-of-day patterns (late night transactions)

---

## ML Tasks for Step 2

### Task 1: Credit Risk — Predict PD (Regression)

**Feature engineering:**
```python
df_loan = pd.read_csv('loan_portfolio.csv')

# Loan-to-income ratio
df_loan['Loan_to_Income'] = df_loan['Loan_Amount'] / df_loan['Annual_Income']

# Payment burden estimate (simplified monthly payment / monthly income)
# Using simple interest approximation
df_loan['Monthly_Payment_Est'] = (df_loan['Loan_Amount'] * (1 + df_loan['Interest_Rate']/100)) / df_loan['Loan_Term_Months']
df_loan['Payment_to_Income'] = df_loan['Monthly_Payment_Est'] / (df_loan['Annual_Income']/12)

# One-hot encode Customer_Type
df_loan = pd.get_dummies(df_loan, columns=['Customer_Type'])
```

**Models:**
- **XGBoost Regressor** — best for tabular data, handles feature interactions
- **Random Forest Regressor** — robust, good feature importance
- **Ridge/Lasso Regression** — baseline, good for understanding linear relationships

**Evaluation metrics for regression:**
- **MAE (Mean Absolute Error)** — average error in PD prediction (e.g., off by 0.03 on average)
- **RMSE** — penalizes large errors more heavily
- **R² score** — proportion of variance explained

**Business output:** Create a risk-scored portfolio table:

| Loan_ID | Predicted_PD | Risk_Bucket | Expected_Loss | Recommendation |
|---|---|---|---|---|
| 13 | 0.28 | Very High | ₹187,735 | Reject |
| 96 | 0.30 | Very High | ₹289,684 | Reject |

### Task 2: Fraud Detection

**Approach 1 — Supervised Classification:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# Engineer features from transactions
df_txn = pd.read_csv('transactions.csv')
df_txn['Timestamp'] = pd.to_datetime(df_txn['Timestamp'])
df_txn['Hour'] = df_txn['Timestamp'].dt.hour
df_txn['DayOfWeek'] = df_txn['Timestamp'].dt.dayofweek

# Customer-level aggregates
customer_stats = df_txn.groupby('Customer_ID')['Amount'].agg(['mean','std','count'])
customer_stats.columns = ['Cust_Avg_Amount', 'Cust_Std_Amount', 'Cust_Txn_Count']
df_txn = df_txn.merge(customer_stats, left_on='Customer_ID', right_index=True)

# Deviation from customer's normal
df_txn['Amount_Zscore'] = (df_txn['Amount'] - df_txn['Cust_Avg_Amount']) / df_txn['Cust_Std_Amount'].clip(lower=1)

# One-hot encode Transaction_Type
df_txn_model = pd.get_dummies(df_txn, columns=['Transaction_Type'])
```

**Key metric: Use Precision-Recall, NOT accuracy.**
- **Precision** = Of all transactions flagged as fraud, how many actually are? (False alarms are costly — they annoy customers)
- **Recall** = Of all actual frauds, how many did we catch? (Missing fraud is very costly)
- **F1 or Average Precision** balances both

**Approach 2 — Unsupervised Anomaly Detection:**
```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Use features that indicate anomaly
features = ['Amount', 'Hour', 'Amount_Zscore', 'Cust_Txn_Count']
X_scaled = StandardScaler().fit_transform(df_txn_model[features])

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_txn['Anomaly_Score'] = iso_forest.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal
```

Isolation Forest works by randomly partitioning data — anomalies are "easy to isolate" (require fewer splits). It's powerful because it doesn't need labels and can catch fraud patterns you didn't anticipate.

### Task 3: Loan Approval Strategy

Combine your PD model with business rules:

```
If Predicted_PD < 0.10: Auto-Approve
If 0.10 <= PD < 0.20: Approve with conditions (higher rate, collateral)
If 0.20 <= PD < 0.25: Manual review
If PD >= 0.25: Auto-Reject
```

Calculate the portfolio-level expected loss: sum of (PD × LGD × Loan_Amount) across all loans. This is the reserve Aurora needs to set aside.

---

# STEP 3: Financial Markets Module — Investment Strategy

*Data not yet provided (market_data.csv). Here are the concepts to study now so you're ready.*

## Key Concepts

### 1. Stock Return Prediction

You'll predict next-day or next-week returns using features like:
- Historical price patterns (moving averages, momentum)
- Volume trends
- Macroeconomic indicators (GDP growth, interest rates)
- Sector-level performance

**Important reality check:** Predicting stock returns is extremely hard. Even a model that's right 52% of the time (barely above random) can be profitable if the wins are larger than the losses. Don't expect high R² values — focus on **directional accuracy** (did the model get the sign of the return right?).

### 2. Portfolio Optimization (Markowitz Mean-Variance)

This is the crown jewel of modern portfolio theory. The idea: **diversification reduces risk without necessarily reducing return.**

**The math:**
- You have `n` assets with expected returns `μ` (vector) and covariance matrix `Σ`
- Portfolio return: `R_p = w'μ` (weighted sum of returns)
- Portfolio risk: `σ_p² = w'Σw` (accounts for correlations)
- The **efficient frontier** is the set of portfolios with maximum return for each level of risk

**Optimization problem:**
```
Minimize: w'Σw (portfolio variance)
Subject to: w'μ = target_return
            sum(w) = 1
            w >= 0 (no short selling, optional)
```

**Python implementation:**
```python
from scipy.optimize import minimize

def portfolio_variance(weights, cov_matrix):
    return weights @ cov_matrix @ weights

def optimize_portfolio(expected_returns, cov_matrix, target_return):
    n = len(expected_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'eq', 'fun': lambda w: w @ expected_returns - target_return}
    ]
    bounds = [(0, 1)] * n  # no short selling
    result = minimize(portfolio_variance, np.ones(n)/n, args=(cov_matrix,),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
```

### 3. Backtesting

Backtesting answers: **if I had used this strategy historically, how would I have performed?**

**Moving window approach:**
1. Train model on data from month 1–12
2. Predict returns for month 13, allocate portfolio
3. Observe actual returns in month 13
4. Slide window: train on months 2–13, predict month 14
5. Repeat

**Metrics to report:**
- **Cumulative return** vs benchmark (e.g., equal-weight portfolio or index)
- **Sharpe Ratio** = (Portfolio_Return - Risk_Free_Rate) / Portfolio_StdDev — higher is better; >1 is good, >2 is excellent
- **Maximum Drawdown** — largest peak-to-trough decline (measures worst-case pain)
- **Sortino Ratio** — like Sharpe but only penalizes downside volatility

### 4. ML for Return Prediction

**Feature engineering for financial time series:**
- **Moving averages:** 5-day, 20-day, 50-day (capture trend)
- **RSI (Relative Strength Index):** Measures overbought/oversold (0–100)
- **Bollinger Bands:** Price relative to rolling mean ± 2 std dev
- **Volume ratio:** Today's volume / 20-day average volume
- **Momentum:** Return over last 5, 10, 20 days
- **Volatility:** Rolling standard deviation of returns
- **Macro features:** GDP growth, interest rate changes

**Models:**
- **XGBoost/Random Forest** — handles non-linear patterns well
- **LASSO Regression** — useful for feature selection (drives unimportant coefficients to zero)
- **Ridge Regression** — good baseline when features are correlated

---

# STEP 4: Derivatives Module — Option Pricing & Hedging

*Data not yet provided (options_data.csv). Here are the concepts to master in advance.*

## Key Concepts

### 1. What Are Options?

An option gives you the **right, but not the obligation**, to buy or sell an asset at a predetermined price.

- **Call option:** Right to BUY at the strike price. Valuable when the stock goes UP.
- **Put option:** Right to SELL at the strike price. Valuable when the stock goes DOWN.

**Key terms:**
- **Strike price (K):** The agreed price at which you can buy/sell
- **Underlying price (S):** Current stock price
- **Expiry (T):** When the option expires
- **Premium:** What you pay to buy the option (this is what you'll predict with ML)

**Payoff at expiry:**
- Call: max(S - K, 0) — you exercise only if the stock is above the strike
- Put: max(K - S, 0) — you exercise only if the stock is below the strike

### 2. Black-Scholes Model

The most famous formula in finance. It prices a European call option:

```
C = S × N(d1) - K × e^(-rT) × N(d2)

where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

N(x) = cumulative standard normal distribution
S = current stock price
K = strike price  
r = risk-free interest rate
T = time to expiry (in years)
σ = volatility of the underlying (annualized)
```

**Python:**
```python
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
```

### 3. The Greeks — Sensitivity Measures

The "Greeks" measure how option price changes when inputs change:

**Delta (Δ):** How much the option price changes per ₹1 change in stock price.
- Call delta: N(d1), ranges from 0 to 1
- Put delta: N(d1) - 1, ranges from -1 to 0
- **Business meaning:** If delta = 0.6, a ₹1 rise in the stock increases the call price by ₹0.60. Delta also approximates the probability of expiring in-the-money.

**Gamma (Γ):** How fast delta itself changes. High gamma means delta is unstable — your hedge needs frequent adjustment.
- Gamma = N'(d1) / (S × σ × √T)
- Highest for at-the-money options near expiry

**Vega (ν):** How much option price changes per 1% change in volatility.
- Vega = S × N'(d1) × √T
- Always positive for both calls and puts
- **Business meaning:** In volatile markets, options become more expensive

**Theta (Θ):** How much the option loses per day just from time passing (time decay).
- Always negative (options lose value as expiry approaches)
- Accelerates near expiry

### 4. Implied Volatility

Black-Scholes takes volatility as input. But in practice, we observe option prices in the market and **back out** the volatility — this is **implied volatility (IV)**.

If the market price differs from Black-Scholes price (using historical volatility), it means the market "implies" a different volatility level. IV captures the market's forward-looking fear/expectations.

**How to compute:** Numerically solve for σ such that BS_price(S, K, T, r, σ) = Market_Price. Use `scipy.optimize.brentq` or Newton's method.

### 5. Delta Hedging

Delta hedging makes a portfolio insensitive to small stock price movements.

**The idea:** If you sold a call option (you're exposed if the stock rises), you can hedge by buying Δ shares of the underlying. As the stock moves, delta changes, so you need to **rebalance** — this is dynamic hedging.

**Simplified example:**
- You sold 100 call options with delta = 0.5
- Your exposure = -100 × 0.5 = -50 shares equivalent
- To hedge: buy 50 shares
- Tomorrow, stock rises, delta becomes 0.6 → you now need 60 shares → buy 10 more

### 6. Value at Risk (VaR)

VaR answers: **What is the maximum loss I can expect over a given time period at a given confidence level?**

- "1-day 95% VaR = ₹5 lakhs" means: on 95% of days, losses won't exceed ₹5 lakhs
- The remaining 5% of days, losses could be worse

**Methods:**
- **Historical VaR:** Sort historical returns, take the 5th percentile
- **Parametric VaR:** Assume returns are normal, VaR = μ - z × σ (where z = 1.645 for 95%)
- **Monte Carlo VaR:** Simulate thousands of scenarios, take the 5th percentile

---

# STEP 5: Integrated Executive Dashboard

This ties everything together. Your dashboard should show:

1. **Project Funding Panel:** Top 5 projects to fund with P(Success), NPV, and risk rating
2. **Credit Risk Panel:** Portfolio risk distribution, expected total loss, high-risk alerts
3. **Fraud Alerts:** Flagged transactions with anomaly scores
4. **Investment Strategy:** Recommended allocation, expected return, Sharpe ratio
5. **Derivatives Risk:** Portfolio VaR, hedging cost, Greek exposures

Use **Plotly** or **Matplotlib/Seaborn** for the visuals. The key is making it executive-friendly — a CFO should be able to glance at it and make decisions.

---

# Workflow Timeline & Practical Roadmap

## Phase 1: NOW (Steps 1 & 2 — data available)

**Week 1–2: Data Exploration**
- Load all three CSVs, compute summary statistics
- Check distributions, correlations, missing values
- Compute NPV, IRR, PI for corporate projects
- Visualize PD distribution across risk factors for loans
- Analyze fraud class imbalance in transactions

**Week 3–4: Model Building**
- Step 1: Classification for project success, regression for cash flows
- Step 2: Regression for PD prediction, classification/anomaly detection for fraud
- Use cross-validation throughout (small datasets!)
- Generate SHAP plots for every model

**Week 5: Business Translation**
- Create ranked project list with recommendations
- Create risk-scored loan portfolio
- Create fraud alert system with monitoring strategy
- Write executive summaries for Steps 1 and 2

## Phase 2: WHEN DATA ARRIVES (Steps 3 & 4)

**Week 6–7: Financial Markets**
- Feature engineering on time series data
- Return prediction models
- Portfolio optimization
- Backtesting

**Week 8–9: Derivatives**
- Black-Scholes pricing and comparison with ML
- Implied volatility computation
- Greeks calculation and delta hedging strategy
- VaR computation

## Phase 3: Integration (Step 5)

**Week 10–11: Dashboard & Presentation**
- Build integrated dashboard
- Create 10–15 slide presentation
- Write executive project brief

---

# Python Libraries You'll Need

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly
pip install scipy statsmodels imbalanced-learn
```

---

# Common Pitfalls to Avoid

1. **Don't chase accuracy on small datasets.** With 50 projects and 100 loans, overfitting is your biggest enemy. Always use cross-validation, and prefer simpler models.

2. **Don't just build models — make decisions.** Every model output should map to a business action: fund/reject, approve/review/deny, buy/sell/hold.

3. **Don't ignore explainability.** A model that says "reject this project" without explaining why is useless to a CFO. SHAP plots and feature importance are not optional — they're 15% of your grade.

4. **Don't treat the steps as independent.** The evaluation gives 20% weight to integration. For example, your portfolio strategy (Step 3) should consider Aurora's lending risk (Step 2) and capital allocation to projects (Step 1).

5. **Don't use accuracy for imbalanced data.** For fraud detection (and potentially credit risk classification), use precision-recall curves, F1, and AUC-ROC.

6. **Don't forget the business context.** When presenting NPV, mention it in the context of Aurora's capital constraints. When presenting PD, mention it in the context of Basel regulatory requirements. This shows you understand finance, not just ML.
