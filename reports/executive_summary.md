# Customer Retention & Churn Intelligence — Executive Briefing

**Project:** Future Interns — Data Science & Analytics, Task 2 (2026)  
**Author:** Data Science & Analytics Team  
**Date:** 2026-11-01  
**Observation Scope:** Verified snapshot of 7,043 subscriber accounts  

---

## 1. Executive Summary

Subscription customer attrition represents both an immediate recurring revenue contraction and a compounding loss of customer lifetime value. This report synthesizes quantitative findings from an end-to-end analytics and machine learning pipeline applied to 7,043 customer accounts.

### Headline Performance Indicators:
- **Total Analyzed Subscriber Base:** **7,043** accounts.
- **Observed Snapshot Churn Prevalence:** **26.54%** (1,869 accounts terminated service).
- **Retention Complement:** **73.46%** (5,174 active customer accounts).
- **Monthly Recurring Billing Exposure:** **$139,130.85** in monthly billings exposed to churn, representing **30.5%** of the organization's total recurring billing volume ($456,116.60).
- **Customer Lifetime Disparity:** Churned accounts have a median tenure of **10.0 months** (mean 18.0 months), compared to **38.0 months** for retained accounts (mean 37.6 months).

---

## 2. Where Attrition Concentrates (Critical Findings)

Our multi-dimensional segmentation, paired with Wilson Score 95% Confidence Intervals, uncovered four primary vulnerability hotspots:

### A. The "Onboarding Drop-Off" (Early Lifecycle Vulnerability)
- **Finding:** Customers in their first 6 months of tenure exhibit a **52.94%** churn prevalence ($N=1,481$; 95% CI: $[50.39\%, 55.47\%]$).
- **Financial Exposure:** Accounts in this 0–6 month window account for **$49,896.10** (35.9%) of all churned monthly revenue exposure.
- **Implication:** The highest risk of customer attrition occurs during initial service activation and habit formation. Drop-off decreases rapidly after month 12 (dropping to 28.7% in months 13–24, and below 10% past month 48).

### B. The Contract Commitment Gap
- **Finding:** Customers on **Month-to-month contracts** experience a **42.71%** churn rate ($N=3,875$; 95% CI: $[41.16\%, 44.27\%]$), compared to **11.27%** for One-Year contracts ($N=1,473$) and **2.83%** for Two-Year contracts ($N=1,695$).
- **Volume Impact:** Month-to-month accounts represent **55.0%** of the total customer base and **88.6%** ($1,655$ out of $1,869$) of all churned accounts.
- **Financial Exposure:** **$120,847.10** in monthly recurring revenue exposure.

### C. Fiber Optic Service Friction & Support Deficit
- **Finding:** **Fiber Optic** internet subscribers experience a **41.89%** churn rate ($N=3,096$; exposure: **$114,300.05$**), contrasted with **18.96%** for DSL subscribers.
- **Root Cause Indicator:** Fiber subscribers **without dedicated Tech Support** churn at **45.8%**, whereas Fiber subscribers with Tech Support churn at only **22.5%**. Fiber customers pay significantly higher monthly charges ($80–$110/mo) and possess heightened reliability expectations.

### D. Friction in Manual Payment Channels
- **Finding:** Subscribers paying via **Electronic Check** experience a **45.29%** churn rate ($N=2,365$; exposure: **$84,288.75$**), compared to **16.71%** for Credit Card auto-pay and **16.76%** for Bank Transfer auto-pay.
- **Implication:** Manual monthly billing requires repeated conscious payment action, amplifying friction and passive involuntary cancellation compared to automated payment methods.

---

## 3. Predictive Modeling & Operational Risk Prioritization

To move from aggregate insights to account-level decision support, we developed and validated a leakage-proof machine learning pipeline.

### Model Benchmarking & Performance
Evaluated across 5-fold Stratified Cross-Validation on training data and tested on 1,409 held-out records:
- **Selected Model:** `LogisticRegression` (with balanced class weights and standardized preprocessing).
- **Held-Out Test ROC-AUC:** **0.8417** (demonstrating strong rank-ordering separation).
- **Held-Out Test PR-AUC:** **0.6331** (vs 0.2654 for zero-skill baseline; a 2.39x precision improvement).
- **Sensitivity / Recall:** **78.34%** (capturing 293 out of 374 churned test accounts).
- **Brier Reliability Score:** **0.1686** (confirming well-calibrated risk probabilities).

### Operational Capacity & Threshold Sweep
Retention teams cannot contact every subscriber. Rather than using an arbitrary 0.50 decision threshold, the model enables dynamic review queue sizing:
- **Threshold 0.60 (High Precision):** Generates a review queue of **372 accounts** (26.4% of base) with **59.9% precision** (capturing 59.6% of churners with minimal wasted contact cost).
- **Threshold 0.50 (Balanced Operations):** Generates a queue of **581 accounts** with **50.4% precision** and **78.3% recall**.
- **Threshold 0.40 (Aggressive Coverage):** Captures **88.2% of churners** ($330$ accounts) at the cost of reviewing **794 accounts** (41.6% precision).

---

## 4. Strategic Retention Action Plan

We recommend deploying four controlled business experiments:

| Strategy | Target Population | Proposed Intervention | Expected Mechanism | Owner | Primary Success Metric | Guardrail Metric |
|---|---|---|---|---|---|---|
| **1. Concierge Onboarding** | New subscribers (tenure $\le 6$ mo) | Proactive 14-day technical setup & check-in call | Resolves early configuration friction and establishes service confidence | Customer Success | 90-day retention rate ($+15\%$ relative lift) | Support cost per onboarded customer |
| **2. Annual Renewal Incentive** | Month-to-month accounts in month 3 | Targeted offer: 1 month free upon 1-year agreement | Extends commitment horizon, eliminating monthly drop-off reflex | Growth Marketing | M2M $\to$ 1-Year plan transition rate ($+20\%$) | Margin dilution / discount cost |
| **3. Fiber Support Trial** | New Fiber Optic installations | 60-day complimentary Tech Support trial | Demonstrates premium service stability during initial trial | Support Operations | 6-month Fiber subscriber retention rate | Support ticket resolution time |
| **4. Auto-Pay Migration** | Active Electronic Check payers | $10 one-time bill credit for ACH / Credit Card auto-pay | Eliminates manual invoice friction and missed payments | Billing Operations | Auto-pay migration rate ($+25\%$) | Payment processing fee variance |

---

## 5. Analytical Limitations & Governance Notice
1. **Retrospective Snapshot:** Metrics reflect a cross-sectional customer state; findings must not be labeled as monthly cohort attrition rates.
2. **Non-Causal Associations:** Predictive model weights represent correlational associations. Interventions must be validated via randomized A/B trials before enterprise-wide rollout.
3. **Privacy & Fair Lending:** Individual identifiers (`customerID`) are excluded from model features; prioritized queues are protected under role-based access.
