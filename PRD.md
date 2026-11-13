# Product Requirements Document (PRD)

## Customer Retention & Churn Intelligence

**Project:** Future Interns --- Data Science & Analytics, Task 2 (2026)\
**Version:** 1.0 \| **Status:** Proposed \| **Primary implementation:**
Python

## 1. Executive summary

Customer Retention & Churn Intelligence is a Python-first analytics
product that helps a subscription business understand customer loss,
identify vulnerable segments, estimate churn risk, and prioritize
evidence-based retention actions.

It combines data-quality validation, descriptive analytics, lifecycle
analysis, predictive modeling, and a stakeholder-facing dashboard. It is
not intended to autonomously make customer decisions. It provides
transparent evidence and prioritization for a human retention team.

The initial release will use a public subscription dataset such as IBM
Telco Customer Churn. The exact source, license, schema, and limitations
must be documented before analysis begins.

## 2. Problem statement

Subscription businesses lose recurring revenue when customers cancel.
Teams may have customer records but lack a consistent view of churn
distribution, vulnerable segments, lifecycle risk, recurring-charge
exposure, customers meriting proactive support, and whether an
intervention improves retention.

A collection of disconnected charts is not sufficient. The product must
define metrics consistently, expose data limitations, and connect
findings to actions that can be evaluated.

## 3. Product vision

Deliver a reproducible, explainable customer-retention analysis
application that helps a product manager or founder move from "what
happened?" to "where should we investigate?" and "what should we test
next?"

## 4. Goals and non-goals

### Goals

1.  Build a validated, reproducible data pipeline.
2.  Provide consistent KPIs for churn, retention, tenure, and revenue
    exposure.
3.  Identify high-churn segments with sample sizes and uncertainty
    context.
4.  Analyze lifecycle patterns and true cohorts when supported by the
    data.
5.  Train and evaluate an interpretable baseline and a candidate model.
6.  Prioritize customers using configurable risk, value, and eligibility
    criteria.
7.  Provide practical recommendations with measurable success criteria.
8.  Make the application runnable locally using documented Python
    commands.

### Non-goals

-   Production customer-facing deployment.
-   Automatic customer contact or subscription changes.
-   Claiming correlations are causal drivers.
-   Claiming churn reduction or revenue savings without an experiment.
-   Real-time streaming for a static sample dataset.
-   Fabricating event timestamps to create calendar cohorts.
-   Claiming a sample dataset represents every SaaS business.

## 5. Users and stakeholder needs

  ------------------------------------------------------------------------
  Persona                 Need                    Product response
  ----------------------- ----------------------- ------------------------
  Founder / business      Understand churn        Executive KPIs and
  owner                   exposure and priorities recommendations

  Product manager         Identify lifecycle and  Segment and tenure
                          service friction        analysis
                          patterns                

  Retention / customer    Prioritize accounts for Risk/value/eligibility
  success                 review                  queue

  Data analyst            Reproduce and validate  Python pipeline,
                          metrics                 definitions, exports

  Reviewer / interviewer  Assess technical and    Documentation, tests,
                          business quality        reproducible outputs
  ------------------------------------------------------------------------

## 6. Assumptions and constraints

-   V1 uses a static customer-level subscription snapshot.
-   Each row represents one customer, subject to validation.
-   The target is a known churn indicator.
-   Python is the primary implementation language.
-   The app is initially local and educational.

**Dataset limitation:** Standard IBM Telco Customer Churn generally
provides tenure but not complete dated subscription events. V1 can
analyze tenure/lifecycle bands but cannot produce defensible
signup-month cohorts or monthly retention curves. Calendar cohorts
require signup dates and dated active/cancellation events.

`MonthlyCharges` may be a recurring-charge exposure proxy if verified;
it is not automatically recognized revenue or MRR. `TotalCharges` is
cumulative and must not be treated as MRR.

**Technical constraints:** fit preprocessing on training data only; keep
a held-out test set untouched until final evaluation; exclude
identifiers and post-outcome fields from predictors; do not publish
restricted raw data.

## 7. Scope and release plan

### V1 --- Required

-   CSV ingestion and schema validation.
-   Data-quality report.
-   Cleaning and type conversion.
-   Descriptive EDA.
-   Customer churn prevalence and segment metrics.
-   Tenure/lifecycle analysis.
-   Python-generated plots and CSV exports.
-   Logistic regression baseline.
-   At least one tree-based comparison model.
-   Evaluation report with appropriate classification metrics.
-   Streamlit dashboard.
-   Evidence-based business recommendations.
-   Tests and documentation.

### V1.1 --- Enhancements

-   Confidence intervals for segment rates.
-   Calibration and threshold/capacity analysis.
-   Permutation importance or SHAP explanations.
-   DuckDB SQL examples.
-   Automated report generation.
-   Model card and data card.

### V2 --- Requires richer data

-   True signup-month cohort retention.
-   Time-to-event survival analysis with valid event/censoring fields.
-   Period-based revenue churn and expansion/contraction.
-   Customer activity features over time.
-   A/B test measurement and intervention uplift.
-   Scheduled scoring and drift monitoring.

## 8. Functional requirements

Priority: **P0** release blocker; **P1** important; **P2** enhancement.

  -----------------------------------------------------------------------
  ID                Priority          Requirement       Acceptance
                                                        criteria
  ----------------- ----------------- ----------------- -----------------
  FR-01             P0                Load CSV from     Missing file
                                      configurable path gives actionable
                                                        error; raw file
                                                        not overwritten

  FR-02             P0                Validate schema   Missing required
                                                        fields reported
                                                        before analysis

  FR-03             P0                Profile data      Rows, duplicates,
                                      quality           nulls, types,
                                                        categories,
                                                        invalid numerics
                                                        exported

  FR-04             P0                Clean safely      Type conversions
                                                        and exclusions
                                                        logged and
                                                        testable

  FR-05             P0                Calculate churn   Numerator,
                                      metrics           denominator,
                                                        target
                                                        definition, scope
                                                        shown

  FR-06             P0                Segment customers Rates include
                                                        group counts and
                                                        can be filtered

  FR-07             P0                Analyze lifecycle Tenure bands and
                                                        churn prevalence
                                                        reported

  FR-08             P0                Train baseline    Reproducible
                                      model             pipeline, fixed
                                                        seed, stratified
                                                        split

  FR-09             P0                Evaluate model    Confusion matrix,
                                                        precision,
                                                        recall, F1,
                                                        ROC-AUC, PR-AUC
                                                        where applicable

  FR-10             P0                Provide dashboard Executive,
                                                        segment,
                                                        lifecycle, model
                                                        views load

  FR-11             P0                Provide           Each
                                      recommendations   recommendation
                                                        cites evidence
                                                        and success
                                                        metric

  FR-12             P0                Document          Static snapshot
                                      limitations       and non-causal
                                                        limits visible

  FR-13             P1                Compare multiple  Same split and
                                      models            metrics for fair
                                                        comparison

  FR-14             P1                Explain model     Feature
                                                        importance
                                                        without causal
                                                        claims

  FR-15             P1                Prioritize risk   Threshold and
                                                        review capacity
                                                        configurable

  FR-16             P1                Export results    Aggregated tables
                                                        and reports
                                                        export to
                                                        documented paths

  FR-17             P2                True cohort       Only enabled with
                                      heatmap           valid signup and
                                                        event dates

  FR-18             P2                SQL analysis      Key aggregates
                                                        reproducible with
                                                        DuckDB SQL
  -----------------------------------------------------------------------

## 9. Data requirements

### Required logical fields

-   Stable customer key.
-   Churn outcome label.
-   Tenure or subscription start/end information.
-   Relevant subscription/service attributes.
-   Price or recurring charge fields if revenue exposure is analyzed.

### Validation rules

-   Customer key is present and unique for a one-row-per-customer
    snapshot.
-   Churn values map to a known binary set.
-   Tenure is numeric and non-negative.
-   Monetary fields are numeric after controlled conversion.
-   Categories are normalized without silently merging distinct
    meanings.
-   Missing values are counted and handled explicitly.
-   Excluded rows are quantified and explained.

### Privacy

Do not include identifiers as model features. Do not publish raw records
or individual risk scores. Avoid sensitive attributes unless necessary,
permitted, and reviewed.

## 10. Metric definitions

Definitions must be centralized in Python functions and displayed
consistently.

  -----------------------------------------------------------------------
  Metric                              Definition / caveat
  ----------------------------------- -----------------------------------
  Dataset churn prevalence            Churned labeled customers / all
                                      labeled customers in snapshot

  Segment churn prevalence            Churned customers in segment /
                                      labeled customers in segment

  Retention complement                1 − churn prevalence, only for same
                                      population and label definition

  Tenure summary                      Median, mean, quartiles, and
                                      observed tenure distribution

  Monthly charge exposure proxy       Sum of `MonthlyCharges` for churned
                                      customers; not confirmed MRR or
                                      realized lost revenue

  Revenue churn                       Requires dated opening and lost
                                      recurring-revenue values for a
                                      defined period

  Cohort retention                    Requires cohort date and dated
                                      active/cancelled status at each age

  CLV                                 Requires stated margin, horizon,
                                      discount rate, and
                                      retention/revenue assumptions
  -----------------------------------------------------------------------

For a static snapshot, do not label churn prevalence as "monthly churn
rate." The source does not define a monthly observation window.

## 11. Modeling requirements

### Target and framing

Predict the provided churn label using features plausibly known at the
chosen scoring time. With a static snapshot, describe this as
retrospective classification unless the data defines a prediction
timestamp and future outcome window.

### Candidate models

1.  `DummyClassifier` baseline.
2.  `LogisticRegression` with suitable preprocessing and class weighting
    where appropriate.
3.  `RandomForestClassifier` or `HistGradientBoostingClassifier`.

### Evaluation

-   Stratified train/test split.
-   Cross-validation on training data.
-   Confusion matrix.
-   Precision, recall, F1.
-   ROC-AUC and average precision / PR-AUC.
-   Calibration curve or Brier score where probabilities are used.
-   Threshold analysis based on intervention capacity and error costs.

Accuracy alone is insufficient for imbalanced classes.

### Leakage controls

Exclude IDs; audit post-cancellation information; fit imputers,
encoders, scalers, and feature selection inside a pipeline; do not tune
on the test set; exclude target-derived fields; record split seed and
model configuration.

## 12. Dashboard requirements

### Page 1 --- Executive overview

-   Total labeled customers.
-   Churned customers.
-   Dataset churn prevalence.
-   Retained customers / retention complement.
-   Monthly-charge exposure proxy, with caveat.
-   Top segments by prevalence and volume.

### Page 2 --- Segment analysis

-   Churn prevalence by contract, tenure band, service, and payment
    method.
-   Group sizes and confidence intervals where feasible.
-   Filters with consistent denominators.

### Page 3 --- Lifecycle / cohorts

-   Tenure distribution and lifecycle bands for V1.
-   Calendar cohort heatmap only with valid dated cohort data.
-   Clear cohort definition.

### Page 4 --- Model evaluation

-   Model comparison.
-   Confusion matrix.
-   Precision-recall and ROC curves.
-   Calibration.
-   Threshold / capacity view.
-   Feature importance.

### Page 5 --- Retention action plan

-   Segment or population.
-   Evidence and sample size.
-   Proposed intervention.
-   Expected mechanism (hypothesis, not established fact).
-   Owner.
-   Success metric.
-   Experiment design and guardrails.

## 13. Non-functional requirements

  -----------------------------------------------------------------------
  Area                                Requirement
  ----------------------------------- -----------------------------------
  Reproducibility                     Fixed seeds, documented
                                      environment, deterministic
                                      transformations

  Maintainability                     Modular Python package and
                                      configuration-driven paths

  Performance                         V1 processes the selected public
                                      dataset on a typical laptop

  Reliability                         Clear validation errors; no silent
                                      data loss

  Usability                           Dashboard labels understandable to
                                      non-technical users

  Explainability                      Model decisions and limitations
                                      documented

  Security                            No secrets, raw personal data, or
                                      restricted data in Git

  Testability                         Unit tests for metrics, cleaning,
                                      features, validation

  Portability                         Setup documented for Windows,
                                      macOS, Linux
  -----------------------------------------------------------------------

## 14. Business recommendation framework

Recommendations are hypotheses to test, not conclusions to assume.

  -----------------------------------------------------------------------
  Evidence to       Intervention      Primary success   Guardrail
  establish         hypothesis        metric            
  ----------------- ----------------- ----------------- -----------------
  Early-tenure band Improve           90-day retention  Support cost per
  has elevated      onboarding and                      retained customer
  churn             early-life                          
                    support                             

  Month-to-month    Test annual-plan  Renewal rate /    Discount cost and
  contracts have    value proposition retained          complaints
  higher prevalence at renewal        recurring revenue 

  Lack of support   Offer proactive   Retention among   Support workload
  add-on correlates help or support   eligible          and satisfaction
  with churn        trial             customers         

  Payment issues    Test payment      Successful        Failed-charge
  correlate with    reminders and     renewal rate      complaints
  churn             retry support                       
  -----------------------------------------------------------------------

Do not assert a pattern until analysis confirms it. Observational data
cannot establish intervention effectiveness.

## 15. Success criteria

### Technical

-   All P0 requirements pass.
-   Data validation and metric tests pass.
-   Pipeline runs from a clean environment.
-   Model evaluation is reproducible.
-   Dashboard loads and displays consistent metrics.
-   Documentation matches implemented behavior.

### Analytical

-   Findings include denominators and sample sizes.
-   At least three meaningful patterns are supported by data.
-   Limitations and uncertainty are stated.
-   Model is compared with a baseline.
-   No unsupported causal claims are made.

### Business communication

-   Recommendations identify target segment, intervention hypothesis,
    owner, and KPI.
-   Executive summary answers "what happened, why investigate, what
    next?"
-   No invented financial impact or model performance is reported.

## 16. Risks and mitigations

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  No dated events         True cohort analysis    Use lifecycle bands;
                          impossible              source event data for
                                                  V2

  Data quality issues     Biased metrics          Validate schema and log
                                                  cleaning

  Class imbalance         Misleading accuracy     PR-AUC, recall,
                                                  precision, threshold
                                                  review

  Leakage                 Inflated test metrics   Pipeline preprocessing
                                                  and feature audit

  Correlation treated as  Poor interventions      Label hypotheses; run
  cause                                           controlled experiments

  Snapshot treated as     Invalid churn rates     State observation scope
  time series                                     and denominator

  Sensitive data exposed  Privacy harm            Exclude raw data and
                                                  row-level scores from
                                                  public repo

  Overfitting             Poor generalization     Cross-validation,
                                                  holdout, baseline
  -----------------------------------------------------------------------

## 17. Delivery plan

1.  Confirm dataset and provenance.
2.  Implement validation and cleaning.
3.  Implement metrics and EDA.
4.  Implement lifecycle analysis.
5.  Implement modeling and evaluation.
6.  Implement dashboard and exports.
7.  Write evidence-based recommendations.
8.  Run tests from a clean environment.
9.  Publish sanitized repository and screenshots.

## 18. Open decisions

-   Final dataset source and license.
-   Whether to retain the Telco snapshot or move to event-level
    subscription data.
-   Whether Streamlit is sufficient or Power BI is also required.
-   Whether survival analysis is justified by available event/censoring
    fields.
-   Which business cost assumptions, if any, are approved for threshold
    selection.

**Document owner:** `<your name>`\
**Review date:** `<YYYY-MM-DD>`\
**Status:** Proposed; revise as implementation decisions are made.
