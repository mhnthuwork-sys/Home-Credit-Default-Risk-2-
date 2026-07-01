# ============================================================
# IFRS 9 Credit Risk Engine
# Python Layer: Data Loading, PD Model, LGD/EAD, ECL, Validation, Stress Test
# Environment: Google Colab
# Dataset: Home Credit Default Risk (Kaggle)
# ============================================================
#
# DATA FILES CAN:
#   - application_train.csv  (307,511 rows x 122 cols)
#   - POS_CASH_balance.csv   (~10M rows x 8 cols)
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

# ============================================================
# STEP 0: CONFIG - chi chinh sua phan nay
# ============================================================

# Duong dan file CSV (chinh lai neu can)
APP_TRAIN_PATH = 'application_train.csv'
POS_CASH_PATH  = 'POS_CASH_balance.csv'
DB_PATH        = 'credit_risk.db'
SQL_PATH       = 'credit_risk.sql'

# Features dung cho PD model (chon tu 122 cot cua application_train)
NUMERIC_FEATURES = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'CNT_CHILDREN', 'REGION_RATING_CLIENT',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]
CATEGORICAL_FEATURES = ['NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE']

print("=" * 60)
print("IFRS 9 Credit Risk Engine - Starting Pipeline")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATA VAO SQLITE
# ============================================================
print("\n[STEP 1] Loading data into SQLite database...")

conn = sqlite3.connect(DB_PATH)

# Load application_train
print("  Loading application_train.csv...")
app = pd.read_csv(APP_TRAIN_PATH)
print(f"  application_train: {app.shape[0]:,} rows x {app.shape[1]} cols")
print(f"  Default rate: {app['TARGET'].mean():.2%}")

# Load POS_CASH_balance (file nay rat lon, in progress message)
print("  Loading POS_CASH_balance.csv (file lon, cho ~30 giay)...")
try:
    pos = pd.read_csv(POS_CASH_PATH)
    print(f"  POS_CASH_balance: {pos.shape[0]:,} rows x {pos.shape[1]} cols")
    has_pos = True
except FileNotFoundError:
    print("  WARNING: POS_CASH_balance.csv khong tim thay.")
    print("  Pipeline se chay khong co phan Staging va Vintage Analysis.")
    pos = pd.DataFrame()
    has_pos = False

# Write vao SQLite
print("  Writing to SQLite...")
app.to_sql('application_train', conn, if_exists='replace', index=False)
if has_pos:
    pos.to_sql('pos_cash_balance', conn, if_exists='replace', index=False)

# Chay SQL file de tao views
print("  Creating views from SQL file...")
with open(SQL_PATH, 'r') as f:
    sql_script = f.read()

# Tach tung statement va chay (bo qua CREATE TABLE da co qua pandas)
for statement in sql_script.split(';'):
    stmt = statement.strip()
    if stmt and ('CREATE VIEW' in stmt.upper() or 'CREATE TABLE IF NOT EXISTS ecl_results' in stmt.upper()):
        try:
            conn.execute(stmt)
        except Exception as e:
            pass  # View co the da ton tai
conn.commit()
print("  [OK] SQLite database ready:", DB_PATH)

# ============================================================
# STEP 2: FEATURE ENGINEERING & PD MODEL
# ============================================================
print("\n[STEP 2] Building PD Model (Logistic Regression Scorecard)...")

# --- 2.1 Chon features ---
features_needed = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['SK_ID_CURR', 'TARGET', 'FLAG_OWN_REALTY']))
df = app[features_needed].copy().reset_index(drop=True)

# --- 2.2 Xu ly missing values ---
for col in NUMERIC_FEATURES:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# --- 2.3 Feature engineering them ---
income_vals = df['AMT_INCOME_TOTAL'].values.astype(float); income_vals[income_vals == 0] = np.nan; ratio_vals = df['AMT_CREDIT'].values / income_vals; ratio_median = float(np.nanmedian(ratio_vals)); ratio_vals[np.isnan(ratio_vals)] = ratio_median; df['CREDIT_INCOME_RATIO'] = ratio_vals

df['AGE_YEARS'] = (-df['DAYS_BIRTH'].values) / 365
df['YEARS_EMPLOYED'] = np.where(df['DAYS_EMPLOYED'].values > 0, 0, -df['DAYS_EMPLOYED'].values / 365)

# --- 2.4 One-hot encoding cho categorical ---
df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

# --- 2.5 Tach X, y ---
exclude_cols = ['SK_ID_CURR', 'TARGET', 'FLAG_OWN_REALTY', 'AMT_CREDIT',
                'DAYS_BIRTH', 'DAYS_EMPLOYED']  # giu lai de dung sau
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

X = df_encoded[feature_cols]
y = df_encoded['TARGET']

print(f"  Features used: {len(feature_cols)}")
print(f"  Class distribution: {y.value_counts().to_dict()}")

# --- 2.6 Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2.7 Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- 2.8 Train Logistic Regression ---
model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- 2.9 Predict PD cho toan bo dataset ---
X_all_scaled = scaler.transform(X[feature_cols])
pd_scores = model.predict_proba(X_all_scaled)[:, 1]

# --- 2.10 Convert sang scorecard points (300-850, kiểu FICO) ---
scorecard_points = 300 + (1 - pd_scores) * 550

print(f"  PD range: {pd_scores.min():.4f} - {pd_scores.max():.4f}")
print(f"  Score range: {scorecard_points.min():.0f} - {scorecard_points.max():.0f}")
print("  [OK] PD Model trained")

# ============================================================
# STEP 3: LGD & EAD
# ============================================================
print("\n[STEP 3] Computing LGD & EAD...")

# LGD: simplified regulatory assumption
# Co bat dong san (FLAG_OWN_REALTY = Y) -> secured -> LGD thap hon
lgd_values = np.where(df['FLAG_OWN_REALTY'] == 'Y', 0.45, 0.75)

# EAD: dung AMT_CREDIT lam proxy
ead_values = df['AMT_CREDIT'].fillna(df['AMT_CREDIT'].median()).values

print(f"  Secured loans (LGD=0.45): {(lgd_values == 0.45).sum():,} ({(lgd_values == 0.45).mean():.1%})")
print(f"  Unsecured loans (LGD=0.75): {(lgd_values == 0.75).sum():,} ({(lgd_values == 0.75).mean():.1%})")
print(f"  Total EAD: {ead_values.sum():,.0f}")
print("  [OK] LGD & EAD computed")

# ============================================================
# STEP 4: ECL CALCULATION (IFRS 9)
# ============================================================
print("\n[STEP 4] Calculating ECL (IFRS 9)...")

# Lay stage tu SQLite neu co pos_cash_balance
if has_pos:
    stage_df = pd.read_sql("""
        SELECT SK_ID_CURR, CURRENT_STAGE
        FROM v_client_current_stage
    """, conn)
    df = df.merge(stage_df, on='SK_ID_CURR', how='left')
    df['CURRENT_STAGE'] = df['CURRENT_STAGE'].fillna(1).astype(int)
else:
    # Gan stage don gian dua tren PD khi khong co pos_cash_balance
    df['CURRENT_STAGE'] = np.where(
        pd_scores > 0.5, 3,
        np.where(pd_scores > 0.2, 2, 1)
    )

stages = df['CURRENT_STAGE'].values

# ECL = PD x LGD x EAD
# Stage 1: ECL 12 thang (PD_12m = PD / 3 la simplification)
# Stage 2 & 3: ECL lifetime (dung full PD)
pd_12m      = pd_scores / 3
pd_lifetime = pd_scores

ecl_values = np.where(
    stages == 1,
    pd_12m * lgd_values * ead_values,          # Stage 1: 12-month ECL
    pd_lifetime * lgd_values * ead_values       # Stage 2 & 3: Lifetime ECL
)

stage_counts = pd.Series(stages).value_counts().sort_index()
print(f"  Stage distribution:")
for s, cnt in stage_counts.items():
    print(f"    Stage {s}: {cnt:,} clients ({cnt/len(stages):.1%})")
print(f"  Total ECL: {ecl_values.sum():,.0f}")
print(f"  ECL / Total EAD: {ecl_values.sum() / ead_values.sum():.2%}")
print("  [OK] ECL calculated")

# ============================================================
# STEP 5: MODEL VALIDATION & STRESS TESTING
# ============================================================
print("\n[STEP 5] Model Validation & Stress Testing...")

# Predict tren tap test
pd_test = model.predict_proba(X_test_scaled)[:, 1]
pd_train = model.predict_proba(X_train_scaled)[:, 1]

# --- AUC ---
auc = roc_auc_score(y_test, pd_test)
print(f"  AUC (Gini = {2*auc-1:.4f}): {auc:.4f}")
print(f"  Benchmark: AUC > 0.70 la acceptable, > 0.75 la good")

# --- Brier Score ---
brier = brier_score_loss(y_test, pd_test)
print(f"  Brier Score: {brier:.4f} (cang thap cang tot, < 0.25 la ok)")

# --- PSI (Population Stability Index) ---
def calculate_psi(expected, actual, buckets=10):
    """
    PSI do su thay doi phan phoi PD giua train va test.
    PSI < 0.10: On dinh, khong can lo
    PSI 0.10-0.25: Co thay doi, can theo doi
    PSI > 0.25: Thay doi lon, can kiem tra lai model
    """
    breakpoints = np.linspace(0, 1, buckets + 1)
    exp_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    # Tranh log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi

psi_value = calculate_psi(pd_train, pd_test)
psi_status = "STABLE" if psi_value < 0.10 else ("MONITOR" if psi_value < 0.25 else "REVIEW NEEDED")
print(f"  PSI: {psi_value:.4f} -> {psi_status}")

# --- Stress Testing ---
print("\n  Stress Test Results:")
print(f"  {'Scenario':<25} {'PD Multiplier':>15} {'Total ECL':>20} {'ECL Change':>12}")
print("  " + "-" * 74)

base_ecl = ecl_values.sum()
stress_scenarios = [
    ("Baseline",         1.0),
    ("Mild Stress",      1.5),
    ("Moderate Stress",  2.0),
    ("Severe Stress",    3.0),
]

stress_results = []
for scenario_name, multiplier in stress_scenarios:
    pd_stressed = np.minimum(pd_scores * multiplier, 1.0)
    ecl_stressed = np.where(
        stages == 1,
        (pd_stressed / 3) * lgd_values * ead_values,
        pd_stressed * lgd_values * ead_values
    )
    total_ecl_stressed = ecl_stressed.sum()
    change_pct = (total_ecl_stressed / base_ecl - 1) * 100
    stress_results.append({
        'Scenario': scenario_name,
        'PD_Multiplier': multiplier,
        'Total_ECL': total_ecl_stressed,
        'ECL_Change_Pct': change_pct
    })
    print(f"  {scenario_name:<25} {multiplier:>15.1f}x {total_ecl_stressed:>20,.0f} {change_pct:>+11.1f}%")

print("  [OK] Validation & stress test complete")

# ============================================================
# STEP 5B: CHARTS
# ============================================================
print("\n[STEP 5B] Generating validation charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IFRS 9 Credit Risk Engine - Model Validation & Portfolio Overview',
             fontsize=14, fontweight='bold', y=0.98)

# Chart 1: ROC Curve
ax1 = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, pd_test)
ax1.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Classifier')
ax1.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve - PD Model Discrimination')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Chart 2: PD Distribution by Target
ax2 = axes[0, 1]
pd_default     = pd_scores[df['TARGET'].values == 1]
pd_non_default = pd_scores[df['TARGET'].values == 0]
ax2.hist(pd_non_default, bins=50, alpha=0.6, color='#16a34a', label='Non-Default (TARGET=0)', density=True)
ax2.hist(pd_default,     bins=50, alpha=0.6, color='#dc2626', label='Default (TARGET=1)',     density=True)
ax2.set_xlabel('Predicted PD')
ax2.set_ylabel('Density')
ax2.set_title('PD Distribution: Default vs Non-Default')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Chart 3: ECL by Stage (bar chart)
ax3 = axes[1, 0]
stage_ecl = {}
for s in [1, 2, 3]:
    mask = stages == s
    stage_ecl[f'Stage {s}'] = ecl_values[mask].sum()

bars = ax3.bar(stage_ecl.keys(), stage_ecl.values(),
               color=['#16a34a', '#f59e0b', '#dc2626'], alpha=0.85, edgecolor='white')
ax3.set_xlabel('IFRS 9 Stage')
ax3.set_ylabel('Total ECL')
ax3.set_title('ECL by IFRS 9 Stage')
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
for bar, val in zip(bars, stage_ecl.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: Stress Test Impact
ax4 = axes[1, 1]
stress_df = pd.DataFrame(stress_results)
colors_stress = ['#16a34a', '#f59e0b', '#f97316', '#dc2626']
bars2 = ax4.bar(stress_df['Scenario'], stress_df['Total_ECL'],
                color=colors_stress, alpha=0.85, edgecolor='white')
ax4.set_xlabel('Stress Scenario')
ax4.set_ylabel('Total ECL')
ax4.set_title('ECL Under Stress Scenarios (PD Multiplier)')
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
for bar, row in zip(bars2, stress_df.itertuples()):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             f'{row.ECL_Change_Pct:+.0f}%', ha='center', va='bottom', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('credit_risk_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Charts saved: credit_risk_charts.png")

# ============================================================
# STEP 6: SAVE RESULTS TO SQLITE & RUN REPORTING QUERIES
# ============================================================
print("\n[STEP 6] Saving results & running reports...")

# --- Luu ECL results vao SQLite ---
ecl_df = pd.DataFrame({
    'SK_ID_CURR':  df['SK_ID_CURR'].values,
    'PD':          pd_scores,
    'LGD':         lgd_values,
    'EAD':         ead_values,
    'ECL':         ecl_values,
    'SCORE':       scorecard_points,
    'IFRS9_STAGE': stages
})
ecl_df.to_sql('ecl_results', conn, if_exists='replace', index=False)

# Tao lai views sau khi co ecl_results
conn.execute("DROP VIEW IF EXISTS report_portfolio_risk")
conn.execute("DROP VIEW IF EXISTS report_scorecard_performance")
conn.execute("DROP VIEW IF EXISTS report_booking_profile")
for statement in sql_script.split(';'):
    stmt = statement.strip()
    if 'CREATE VIEW IF NOT EXISTS report_' in stmt:
        try:
            conn.execute(stmt)
        except Exception as e:
            pass
conn.commit()

# --- R1: Portfolio Risk Report ---
print("\n  R1: PORTFOLIO RISK REPORT")
print("  " + "=" * 70)
r1 = pd.read_sql("SELECT * FROM report_portfolio_risk", conn)
print(r1.to_string(index=False))

# --- R2: Scorecard Performance Report ---
print("\n  R2: SCORECARD PERFORMANCE REPORT")
print("  " + "=" * 70)
r2 = pd.read_sql("SELECT * FROM report_scorecard_performance", conn)
print(r2.to_string(index=False))

# --- R3: Booking Profile ---
print("\n  R3: BOOKING PROFILE")
print("  " + "=" * 70)
r3 = pd.read_sql("SELECT * FROM report_booking_profile", conn)
print(r3.to_string(index=False))

conn.close()

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Database:  {DB_PATH}")
print(f"  Charts:    credit_risk_charts.png")
print(f"  Tables:    application_train, pos_cash_balance, ecl_results")
print(f"  Views:     v_ifrs9_staging, v_client_current_stage,")
print(f"             v_vintage_analysis, v_exposure,")
print(f"             report_portfolio_risk, report_scorecard_performance,")
print(f"             report_booking_profile")
print("\n  KEY RESULTS:")
print(f"    AUC:        {auc:.4f}  (Gini = {2*auc-1:.4f})")
print(f"    Brier:      {brier:.4f}")
print(f"    PSI:        {psi_value:.4f} ({psi_status})")
print(f"    Total EAD:  {ead_values.sum():>20,.0f}")
print(f"    Total ECL:  {ecl_values.sum():>20,.0f}")
print(f"    Coverage:   {ecl_values.sum()/ead_values.sum():.2%}")
