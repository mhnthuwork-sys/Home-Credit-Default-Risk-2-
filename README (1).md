# IFRS 9 Credit Risk Engine
## PD-LGD-EAD Modelling, Validation & Portfolio Reporting

> **Portfolio project** simulating the end-to-end workflow of a credit risk pipline, aligned with **Basel II/III** and **IFRS 9** standards.

---

## Project Overview

This project builds a complete credit risk modelling pipeline on the **Home Credit Default Risk** dataset (Kaggle, 307,511 retail loan applications). It covers all major components a credit risk process works on: model development, IFRS 9 staging, ECL calculation, model validation, stress testing, and portfolio reporting.

### Why This Project?

Designed to cover:
- PD/LGD/EAD model development per Basel II/IFRS 9
- Model validation (discrimination, calibration, stability)
- Portfolio stress testing
- Risk reporting for senior management and regulators

---

## Data Flow

```
RAW DATA (Kaggle Home Credit)
│
├── application_train.csv         POS_CASH_balance.csv
│   (307,511 rows x 122 cols)     (~10M rows x 8 cols)
│   1 row per client              Monthly DPD history per loan
│   TARGET = 1 if default         SK_DPD = Days Past Due
│
└─────────────────┬───────────────────────┘
                  │ Join on SK_ID_CURR
                  ▼
        ┌─────────────────────┐
        │  STEP 1: SQL DATAMART│  (SQLite via sqlite3)
        │  - IFRS 9 Staging   │  LAG(SK_DPD) -> Stage 1/2/3
        │  - Vintage Analysis │  Cohort cumulative default rate
        │  - Exposure View    │  EAD = AMT_CREDIT, LGD by collateral
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  STEP 2: PD MODEL   │  (sklearn LogisticRegression)
        │  Features: EXT_SOURCE│  WOE-style risk signals
        │  Output: PD (0-1)   │  + Scorecard (300-850 pts)
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  STEP 3: LGD & EAD  │  Simplified regulatory assumption
        │  LGD = 0.45/0.75    │  Based on collateral (FLAG_OWN_REALTY)
        │  EAD = AMT_CREDIT   │  Exposure at Default proxy
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  STEP 4: ECL (IFRS9)│
        │  Stage 1: PD_12m    │  ECL = PD × LGD × EAD
        │  Stage 2/3: PD_life │  Lifetime vs 12-month horizon
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  STEP 5: VALIDATION │
        │  AUC, Brier, PSI    │  Discrimination + Calibration
        │  Stress Test        │  PD x 1.5 / 2.0 / 3.0
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  STEP 6: REPORTING  │  (SQL Views)
        │  Portfolio Risk     │  ECL by Stage, Coverage Ratio
        │  Scorecard Perf.    │  Score gap: Default vs Non-Default
        │  Booking Profile    │  Distribution by Score Band x Stage
        └─────────────────────┘
```

---

## Tools & Libraries

| Layer        | Tool / Library              | Purpose                                      |
|--------------|-----------------------------|----------------------------------------------|
| Database     | SQLite (built-in `sqlite3`) | Datamart, staging, vintage, reporting views  |
| Data wrangling | `pandas`, `numpy`         | Feature engineering, ECL calculation         |
| Modelling    | `scikit-learn`              | Logistic Regression, train/test split, scaler|
| Validation   | `sklearn.metrics`           | AUC, Brier Score; custom PSI function        |
| Visualisation| `matplotlib`                | ROC curve, PD distribution, ECL charts       |
| Environment  | Google Colab                | Free GPU/CPU, no local setup needed          |

No additional installs required in Google Colab — all libraries are pre-installed.

---

## File Structure

```
credit_risk_project/
├── credit_risk.py       # Main pipeline: Steps 0-6 (run top to bottom)
├── credit_risk.sql      # SQL: CREATE TABLE, Views, Reporting queries
├── README.md            # This file
│
├── [auto-generated on run]
│   ├── credit_risk.db          # SQLite database with all tables & views
│   └── credit_risk_charts.png  # Validation charts (4-panel)
│
└── [data files — not included, download from Kaggle]
    ├── application_train.csv
    └── POS_CASH_balance.csv

---

## Key Results Interpretation

### AUC (Area Under ROC Curve)
- Measures how well the model separates defaulters from non-defaulters
- **> 0.70**: Acceptable for regulatory use
- **> 0.75**: Good
- **> 0.80**: Excellent
- Related: **Gini coefficient** = 2 × AUC − 1 (common in Basel reporting)

### Brier Score
- Measures accuracy of predicted probabilities (not just ranking)
- **< 0.25**: Generally acceptable
- **Lower is better** (0 = perfect, 0.25 = no skill)

### PSI (Population Stability Index)
- Measures whether the PD distribution has shifted between train and test
- **< 0.10**: Stable — model can be used as-is
- **0.10–0.25**: Monitor — investigate feature drift
- **> 0.25**: Unstable — model needs recalibration or rebuild

### ECL Coverage Ratio = Total ECL / Total EAD
- The percentage of total exposure covered by expected loss provisions
- Reported by Stage (Stage 3 coverage should be highest)

### IFRS 9 Stages
| Stage | DPD Condition | ECL Horizon | Typical Action |
|-------|--------------|-------------|----------------|
| 1     | DPD ≤ 30 days | 12-month | Standard provisioning |
| 2     | DPD 31–90 days OR significant increase in credit risk | Lifetime | Increased monitoring |
| 3     | DPD > 90 days | Lifetime | Non-performing, collections |

---

## Assumptions & Limitations

### LGD Simplification
LGD in this project uses a **rule-based proxy**:
- `FLAG_OWN_REALTY = 'Y'` (secured by real estate) → LGD = 0.45
- `FLAG_OWN_REALTY = 'N'` (unsecured) → LGD = 0.75

**Limitation**: True LGD should be estimated from workout data (actual recoveries post-default). The Home Credit dataset does not provide post-default recovery cash flows, so this approach uses regulatory-style LGD floors as a simplification. In practice, LGD models use regression on recovery rates, cure rates, and collateral valuations.

### PD Model Scope
- Model is trained on **application-level** features (PIT PD at origination)
- Does not incorporate behavioural features from POS_CASH_balance (which would produce a more accurate behavioural score)
- PD calibration to long-run average (TTC PD) is not performed — this would require multi-year default rate history

### EAD Simplification
- EAD = AMT_CREDIT (outstanding principal at application)
- In practice, EAD for revolving facilities (credit cards) requires CCF (Credit Conversion Factor) modelling to account for undrawn limits

### ECL 12-Month Approximation
- Stage 1 ECL uses `PD_12m = PD / 3` as a simplification
- In practice, 12-month PD is estimated from the survival function of the PD model over a 12-month horizon

### Dataset Vintage Limitation
- Home Credit provides a cross-sectional snapshot; true panel-based vintage curves require cohort tracking over multiple years of performance data

---

## Key References

- IFRS 9 Financial Instruments: IASB Standard (2014)
- Basel II/III: BCBS Framework for Credit Risk (2006, 2017)
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Home Credit Default Risk: https://www.kaggle.com/c/home-credit-default-risk

## Key References

- IFRS 9 Financial Instruments: IASB Standard (2014)
- Basel II/III: BCBS Framework for Credit Risk (2006, 2017)
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Home Credit Default Risk: https://www.kaggle.com/c/home-credit-default-risk
