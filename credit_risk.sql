-- ============================================================
-- IFRS 9 Credit Risk Engine
-- SQL Layer: Datamart, Staging, Vintage, Exposure, Reporting
-- Engine: SQLite 3.25+ (supports CTEs and Window Functions)
-- Dataset: Home Credit Default Risk (Kaggle)
-- ============================================================

-- ============================================================
-- STEP 1A: CREATE BASE TABLES
-- (Chay sau khi Python da load CSV vao SQLite bang sqlite3)
-- ============================================================

-- Bang chinh: thong tin khach hang tai thoi diem apply vay
CREATE TABLE IF NOT EXISTS application_train (
    SK_ID_CURR          INTEGER PRIMARY KEY,
    TARGET              INTEGER,           -- 1 = default, 0 = no default
    NAME_CONTRACT_TYPE  TEXT,
    CODE_GENDER         TEXT,
    FLAG_OWN_CAR        TEXT,
    FLAG_OWN_REALTY     TEXT,              -- Y = co bat dong san (dung tinh LGD)
    CNT_CHILDREN        INTEGER,
    AMT_INCOME_TOTAL    REAL,
    AMT_CREDIT          REAL,              -- EAD proxy
    AMT_ANNUITY         REAL,
    AMT_GOODS_PRICE     REAL,
    NAME_EDUCATION_TYPE TEXT,
    NAME_FAMILY_STATUS  TEXT,
    NAME_INCOME_TYPE    TEXT,
    NAME_HOUSING_TYPE   TEXT,
    DAYS_BIRTH          INTEGER,           -- so ngay am, tinh tuoi = -DAYS_BIRTH/365
    DAYS_EMPLOYED       INTEGER,           -- so ngay am, tinh nam di lam = -DAYS_EMPLOYED/365
    DAYS_REGISTRATION   REAL,
    DAYS_ID_PUBLISH     INTEGER,
    OCCUPATION_TYPE     TEXT,
    CNT_FAM_MEMBERS     REAL,
    REGION_RATING_CLIENT INTEGER,
    EXT_SOURCE_1        REAL,              -- diem tin dung ngoai (0-1, cang cao cang tot)
    EXT_SOURCE_2        REAL,
    EXT_SOURCE_3        REAL,
    OBS_30_CNT_SOCIAL_CIRCLE  REAL,
    DEF_30_CNT_SOCIAL_CIRCLE  REAL,
    OBS_60_CNT_SOCIAL_CIRCLE  REAL,
    DEF_60_CNT_SOCIAL_CIRCLE  REAL,
    AMT_REQ_CREDIT_BUREAU_YEAR REAL
);

-- Bang panel: lich su tra no theo thang cua tung khoan vay
CREATE TABLE IF NOT EXISTS pos_cash_balance (
    SK_ID_PREV              INTEGER,       -- ma khoan vay truoc do
    SK_ID_CURR              INTEGER,       -- ma khach hang (join key)
    MONTHS_BALANCE          INTEGER,       -- thang tuong doi (0 = hien tai, -1 = 1 thang truoc...)
    CNT_INSTALMENT          REAL,
    CNT_INSTALMENT_FUTURE   REAL,
    NAME_CONTRACT_STATUS    TEXT,
    SK_DPD                  INTEGER,       -- Days Past Due: so ngay qua han thang nay
    SK_DPD_DEF              INTEGER        -- DPD sau khi tru tolerance
);

-- ============================================================
-- STEP 1B: IFRS 9 STAGING TABLE
-- Logic:
--   Stage 1: SK_DPD <= 30 (binh thuong, ECL 12 thang)
--   Stage 2: SK_DPD 31-90 HOAC tang dot bien so voi thang truoc (rui ro tang, ECL lifetime)
--   Stage 3: SK_DPD > 90 (da default, ECL lifetime)
-- Dung LAG() de so sanh DPD thang nay vs thang truoc -> phat hien "significant increase in credit risk"
-- ============================================================

CREATE VIEW IF NOT EXISTS v_ifrs9_staging AS
WITH monthly_dpd AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        SK_DPD,
        -- LAG: lay DPD cua thang truoc de so sanh
        LAG(SK_DPD, 1, 0) OVER (
            PARTITION BY SK_ID_PREV
            ORDER BY MONTHS_BALANCE
        ) AS SK_DPD_PREV_MONTH,
        NAME_CONTRACT_STATUS
    FROM pos_cash_balance
),
staged AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        SK_DPD,
        SK_DPD_PREV_MONTH,
        CASE
            WHEN SK_DPD > 90
                THEN 3  -- Stage 3: default/credit impaired
            WHEN SK_DPD BETWEEN 31 AND 90
                THEN 2  -- Stage 2: significant increase in credit risk
            WHEN (SK_DPD - SK_DPD_PREV_MONTH) >= 30
                THEN 2  -- Stage 2: DPD tang dot bien >= 30 ngay so voi thang truoc
            ELSE 1      -- Stage 1: binh thuong
        END AS IFRS9_STAGE
    FROM monthly_dpd
)
SELECT
    s.*,
    -- Lay stage moi nhat (MONTHS_BALANCE = 0 hoac lon nhat) cho moi khoan vay
    MAX(MONTHS_BALANCE) OVER (PARTITION BY SK_ID_PREV) AS LATEST_MONTH
FROM staged s;

-- Stage hien tai (moi nhat) cua tung khach hang
CREATE VIEW IF NOT EXISTS v_client_current_stage AS
SELECT
    SK_ID_CURR,
    -- Neu khach co nhieu khoan vay, lay stage xau nhat (cao nhat)
    MAX(IFRS9_STAGE) AS CURRENT_STAGE,
    MAX(SK_DPD)      AS MAX_DPD,
    COUNT(DISTINCT SK_ID_PREV) AS NUM_LOANS
FROM v_ifrs9_staging
WHERE MONTHS_BALANCE = LATEST_MONTH
GROUP BY SK_ID_CURR;

-- ============================================================
-- STEP 1C: VINTAGE ANALYSIS VIEW
-- Vintage = nhom cac khoan vay duoc giai ngan cung thoi diem,
-- theo doi ty le default tang dan theo so thang on-book (MOB)
-- Dung de nhin "khoan vay cung cot" xu huong xau di the nao theo thoi gian
-- ============================================================

CREATE VIEW IF NOT EXISTS v_vintage_analysis AS
WITH cohort_base AS (
    SELECT
        SK_ID_PREV,
        SK_ID_CURR,
        -- Thang giai ngan = thang dau tien xuat hien trong pos_cash_balance
        MIN(MONTHS_BALANCE) AS DISBURSEMENT_MONTH
    FROM pos_cash_balance
    GROUP BY SK_ID_PREV, SK_ID_CURR
),
cohort_monthly AS (
    SELECT
        cb.DISBURSEMENT_MONTH                          AS COHORT,
        p.MONTHS_BALANCE - cb.DISBURSEMENT_MONTH       AS MOB,  -- Months On Book
        COUNT(DISTINCT p.SK_ID_PREV)                   AS TOTAL_LOANS,
        SUM(CASE WHEN p.SK_DPD > 90 THEN 1 ELSE 0 END) AS DEFAULT_COUNT
    FROM pos_cash_balance p
    JOIN cohort_base cb
        ON p.SK_ID_PREV = cb.SK_ID_PREV
    GROUP BY cb.DISBURSEMENT_MONTH, MOB
)
SELECT
    COHORT,
    MOB,
    TOTAL_LOANS,
    DEFAULT_COUNT,
    ROUND(1.0 * DEFAULT_COUNT / NULLIF(TOTAL_LOANS, 0) * 100, 2) AS DEFAULT_RATE_PCT
FROM cohort_monthly
ORDER BY COHORT, MOB;

-- ============================================================
-- STEP 1D: EXPOSURE VIEW (EAD)
-- EAD proxy = AMT_CREDIT tu application_train
-- LGD rule:
--   FLAG_OWN_REALTY = 'Y' (co tai san dam bao bat dong san) -> LGD = 0.45
--   FLAG_OWN_REALTY = 'N' (khong co tai san) -> LGD = 0.75
-- Day la simplified assumption, ghi ro trong README
-- ============================================================

CREATE VIEW IF NOT EXISTS v_exposure AS
SELECT
    SK_ID_CURR,
    AMT_CREDIT                                    AS EAD,
    CASE
        WHEN FLAG_OWN_REALTY = 'Y' THEN 0.45
        ELSE 0.75
    END                                           AS LGD,
    ROUND(-DAYS_BIRTH / 365.0, 1)                 AS AGE,
    ROUND(-DAYS_EMPLOYED / 365.0, 1)              AS YEARS_EMPLOYED,
    AMT_INCOME_TOTAL,
    ROUND(AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0), 2) AS CREDIT_INCOME_RATIO
FROM application_train;

-- ============================================================
-- STEP 6: REPORTING LAYER
-- Cac query nay chay sau khi Python da tinh PD va ECL,
-- ket qua duoc luu vao bang ecl_results
-- ============================================================

-- Bang luu ket qua ECL tu Python (Python se CREATE va INSERT vao day)
CREATE TABLE IF NOT EXISTS ecl_results (
    SK_ID_CURR    INTEGER PRIMARY KEY,
    PD            REAL,    -- Probability of Default (0-1)
    LGD           REAL,    -- Loss Given Default
    EAD           REAL,    -- Exposure at Default
    ECL           REAL,    -- Expected Credit Loss
    SCORE         REAL,    -- Scorecard points (300-850)
    IFRS9_STAGE   INTEGER  -- 1, 2, hoac 3
);

-- R1: Portfolio Risk Report - tong quan rui ro toan danh muc theo Stage
-- KPI chinh ma senior management va regulator quan tam
CREATE VIEW IF NOT EXISTS report_portfolio_risk AS
SELECT
    IFRS9_STAGE                             AS Stage,
    COUNT(*)                                AS Num_Clients,
    ROUND(SUM(EAD), 0)                      AS Total_EAD,
    ROUND(SUM(ECL), 0)                      AS Total_ECL,
    ROUND(AVG(PD) * 100, 2)                 AS Avg_PD_Pct,
    ROUND(AVG(LGD) * 100, 2)               AS Avg_LGD_Pct,
    ROUND(SUM(ECL) / NULLIF(SUM(EAD), 0) * 100, 2) AS ECL_Coverage_Ratio_Pct
FROM ecl_results
GROUP BY IFRS9_STAGE
ORDER BY IFRS9_STAGE;

-- R2: Scorecard Performance Report - phan biet default vs non-default qua diem
-- Dung de validate scorecard co phan biet tot khong (gap diem cang lon cang tot)
CREATE VIEW IF NOT EXISTS report_scorecard_performance AS
SELECT
    CASE WHEN er.IFRS9_STAGE = 3 THEN 'Default' ELSE 'Non-Default' END AS Client_Group,
    COUNT(*)                                AS Num_Clients,
    ROUND(AVG(er.SCORE), 1)                AS Avg_Score,
    ROUND(MIN(er.SCORE), 1)                AS Min_Score,
    ROUND(MAX(er.SCORE), 1)                AS Max_Score,
    ROUND(AVG(er.PD) * 100, 2)             AS Avg_PD_Pct
FROM ecl_results er
GROUP BY Client_Group;

-- R3: Booking Profile - phan bo khach hang theo band diem va stage
-- Gup nhin full picture cua danh muc
CREATE VIEW IF NOT EXISTS report_booking_profile AS
SELECT
    CASE
        WHEN SCORE BETWEEN 300 AND 449 THEN '300-449 (High Risk)'
        WHEN SCORE BETWEEN 450 AND 599 THEN '450-599 (Medium Risk)'
        WHEN SCORE BETWEEN 600 AND 749 THEN '600-749 (Low Risk)'
        WHEN SCORE BETWEEN 750 AND 850 THEN '750-850 (Very Low Risk)'
        ELSE 'Out of Range'
    END                     AS Score_Band,
    IFRS9_STAGE             AS Stage,
    COUNT(*)                AS Num_Clients,
    ROUND(SUM(EAD), 0)      AS Total_EAD,
    ROUND(SUM(ECL), 0)      AS Total_ECL
FROM ecl_results
GROUP BY Score_Band, IFRS9_STAGE
ORDER BY Score_Band, IFRS9_STAGE;
