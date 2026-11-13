# Data Quality & Schema Profiling Report

**Generated:** 2026-11-13T06:15:43.250310+00:00 (UTC)  
**Source File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**SHA-256 Fingerprint:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`  
**Status:** ✅ PASSED ALL GATES

## 1. Executive Quality Summary

- **Total Rows:** 7,043
- **Total Columns:** 21
- **Unique Customer Keys:** 7,043
- **Schema Gate:** PASSED
- **Row Grain Gate:** PASSED
- **Target Gate:** PASSED

## 2. Target Variable Distribution

| Churn Label | Record Count | Percentage |
|---|---|---|
| **No** | 5,174 | 73.46% |
| **Yes** | 1,869 | 26.54% |

## 3. Data Integrity & Quirks Audit

- **Tenure Range:** 0 to 72 months
- **Monthly Charges Range:** $18.25 to $118.75
- **TotalCharges Whitespace Count:** 11 records

### Audit of Whitespace TotalCharges:
As established in the data specification, exactly 11 customers have `tenure == 0` (new enrollments who joined mid-billing cycle). Their raw `TotalCharges` strings are whitespace `' '`. The cleaning pipeline will safely coerce these to `0.0`.

## 4. Warnings and Exceptions

- ⚠️ **WARNING:** Found 11 records where TotalCharges is whitespace string ' '. Documented cleaning policy will impute 0.0 for tenure==0.
