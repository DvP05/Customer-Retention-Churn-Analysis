# Data Directory & Provenance

## 1. Dataset Overview
- **Dataset:** IBM Telco Customer Churn
- **Standard Filename:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Source Mirror:** Official IBM Community Dataset Repository (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`)
- **Original Source Attribution:** IBM Sample Data / Blastchar Kaggle Distribution (`https://www.kaggle.com/datasets/blastchar/telco-customer-churn`)
- **License:** Apache License 2.0 / Public Domain / Educational & Research Use
- **Retrieval Date:** 2026-11-01
- **File Format:** Comma-Separated Values (CSV)
- **File Size:** ~977,501 bytes
- **Record Count:** 7,043 rows
- **Attribute Count:** 21 columns
- **SHA-256 Checksum:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

---

## 2. Data Grain & Observation Scope
- **Grain:** Exactly one row per unique customer account (`customerID`).
- **Nature of Data:** Cross-sectional retrospective customer snapshot.
- **Critical Limitation:** Does **not** contain individual transaction event logs or dated subscription start/end timestamps. It supports **lifecycle tenure-band analysis** but does **not** support true calendar cohort curves or monthly attrition rates without fabricated assumptions.

---

## 3. Data Dictionary

| Column Name | Data Type | Description | Observed Categories / Value Range |
|---|---|---|---|
| `customerID` | `string` | Unique customer alphanumeric identifier | 7,043 distinct keys (e.g., `7590-VHVEG`) |
| `gender` | `string` | Customer gender identity | `Female`, `Male` |
| `SeniorCitizen` | `int64` / `category` | Indicator whether customer is senior citizen | `0` (No), `1` (Yes) |
| `Partner` | `string` | Whether customer has a domestic partner | `Yes`, `No` |
| `Dependents` | `string` | Whether customer has financial dependents | `Yes`, `No` |
| `tenure` | `int64` | Total months customer has maintained account | `0` to `72` months |
| `PhoneService` | `string` | Whether customer subscribes to home phone | `Yes`, `No` |
| `MultipleLines` | `string` | Multiple phone lines subscription | `No`, `Yes`, `No phone service` |
| `InternetService` | `string` | Primary internet connection technology | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | `string` | Cyber security protection add-on | `No`, `Yes`, `No internet service` |
| `OnlineBackup` | `string` | Cloud backup storage add-on | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | `string` | Hardware warranty & device replacement add-on | `No`, `Yes`, `No internet service` |
| `TechSupport` | `string` | Dedicated priority technical assistance add-on | `No`, `Yes`, `No internet service` |
| `StreamingTV` | `string` | Television streaming entertainment add-on | `No`, `Yes`, `No internet service` |
| `StreamingMovies` | `string` | On-demand movie streaming add-on | `No`, `Yes`, `No internet service` |
| `Contract` | `string` | Contract billing agreement term | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | `string` | Digital paperless invoicing opt-in | `Yes`, `No` |
| `PaymentMethod` | `string` | Selected invoice settlement method | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | `float64` | Current monthly recurring billing charge ($) | `$18.25` to `$118.75` |
| `TotalCharges` | `string` $\to$ `float64` | Cumulative billing charges over customer life | `$18.80` to `$8684.80` (11 rows with `" "` for tenure 0) |
| `Churn` | `string` | Customer cancellation indicator (target label) | `No` (5,174; 73.46%), `Yes` (1,869; 26.54%) |

---

## 4. Known Data Quirks & Handling Policy
1. **`TotalCharges` Whitespace Sentinels:** 11 records have `" "` due to `tenure == 0` (new accounts who enrolled mid-cycle). In `clean.py`, these are imputed to `0.0` with explicit auditing.
2. **Dependent Service Levels:** Service add-on fields contain `"No internet service"` or `"No phone service"` indicating service absence rather than an active opt-out.
