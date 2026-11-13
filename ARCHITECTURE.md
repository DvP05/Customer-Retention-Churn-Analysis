# Technical Architecture

## Customer Retention & Churn Intelligence

**Project:** Future Interns --- Data Science & Analytics, Task 2 (2026)\
**Version:** 1.0 \| **Architecture:** Modular Python analytics
application\
**Deployment target:** Local development / portfolio demonstration

## 1. Architecture overview

The solution is a modular, batch-oriented Python application. Raw data
enters through controlled ingestion, is validated and transformed, then
feeds analytical and modeling paths. Outputs are persisted as aggregate
tables, figures, metrics, and a trained model. A Streamlit application
reads these artifacts and presents them to stakeholders.

V1 does not require a web backend, cloud infrastructure, message queue,
or database server. DuckDB may be added for SQL analysis without
introducing a database service.

## 2. Architecture principles

1.  **Python-first:** ingestion, cleaning, metrics, modeling,
    visualization, and dashboard logic are primarily Python.
2.  **Reproducible:** fixed seeds, configuration-driven paths,
    documented environment, deterministic transformations.
3.  **Raw-data immutability:** never overwrite the original source.
4.  **Separation of concerns:** ingestion, validation, cleaning,
    analysis, modeling, and presentation are separate modules.
5.  **Metric consistency:** reports and dashboard call the same metric
    functions.
6.  **Leakage prevention:** preprocessing is fitted inside training
    pipelines.
7.  **Honest time semantics:** snapshot metrics are not mislabeled as
    monthly rates; cohorts require dated events.
8.  **Human oversight:** risk scores inform investigation; they do not
    trigger automatic customer actions.
9.  **Privacy by default:** identifiers and row-level predictions are
    not published.

## 3. High-level component diagram

``` mermaid
flowchart TD
    A[Public subscription CSV] --> B[Ingestion layer]
    C[Config and schema definitions] --> B
    B --> D[Data validation and profiling]
    D -->|Valid| E[Cleaning and normalization]
    D -->|Invalid| X[Validation report and failure]
    E --> F[Processed analytical dataset]
    F --> G[Descriptive analytics]
    F --> H[Lifecycle / cohort analysis]
    F --> I[Feature engineering]
    I --> J[Train / validation / test workflow]
    J --> K[Model evaluation and explanation]
    G --> L[Reports, tables and figures]
    H --> L
    K --> M[Model artifact and metrics]
    L --> N[Streamlit dashboard]
    M --> N
    N --> O[Stakeholder insights and action hypotheses]
```

## 4. Runtime and deployment view

``` mermaid
flowchart LR
    subgraph LocalMachine[Developer or reviewer laptop]
      CLI[Python CLI / scripts]
      DATA[(Local data folders)]
      PKG[retention Python package]
      ART[(Reports and model artifacts)]
      APP[Streamlit application]
      TEST[pytest suite]
      CLI --> PKG
      PKG <--> DATA
      PKG --> ART
      APP --> ART
      TEST --> PKG
    end
    USER[Analyst / reviewer] --> APP
```

V1 runs locally. If deployed later, add authentication, protected
storage, access logging, secrets management, and a privacy/security
review before using real customer data.

## 5. Repository and module boundaries

``` text
src/retention/
├── config.py
├── data/
│   ├── load.py
│   ├── validate.py
│   └── clean.py
├── analysis/
│   ├── metrics.py
│   ├── cohorts.py
│   ├── segmentation.py
│   └── lifetime.py
├── features/
│   └── build.py
├── modeling/
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── reporting/
│   └── charts.py
└── app/
    └── streamlit_app.py
```

  ----------------------------------------------------------------------------------
  Module                       Responsibility    Input             Output
  ---------------------------- ----------------- ----------------- -----------------
  `config.py`                  Load paths,       YAML/defaults     Configuration
                               target, columns,                    
                               seeds, model                        
                               settings                            

  `data/load.py`               Read source and   CSV path          Raw DataFrame
                               record metadata                     

  `data/validate.py`           Check schema,     Raw DataFrame     Validation
                               keys, ranges,                       report/errors
                               categories, nulls                   

  `data/clean.py`              Normalize and     Validated         Clean DataFrame
                               convert values    DataFrame         

  `analysis/metrics.py`        Canonical         Clean DataFrame   KPI tables
                               churn/retention                     
                               measures                            

  `analysis/segmentation.py`   Segment rates and Clean DataFrame   Segment summaries
                               comparisons                         

  `analysis/cohorts.py`        Tenure bands or   Clean/event data  Cohort table
                               true dated                          
                               cohorts                             

  `analysis/lifetime.py`       Tenure summaries  Valid             Lifetime outputs
                               and supported     duration/event    
                               survival analysis fields            

  `features/build.py`          Define predictors Clean DataFrame   X, y, metadata
                               and target                          

  `modeling/train.py`          Split, fit,       X, y              Model artifact
                               cross-validate,                     
                               select                              

  `modeling/evaluate.py`       Evaluate held-out Model and test    Metrics/plots
                               predictions       data              

  `modeling/explain.py`        Permutation       Fitted model,     Explanations
                               importance /      test data         
                               optional SHAP                       

  `reporting/charts.py`        Standardized      Analysis tables   PNG/HTML
                               figures                             

  `app/streamlit_app.py`       Present metrics,  Saved artifacts   Dashboard
                               charts, filters                     
  ----------------------------------------------------------------------------------

Keep reusable logic in importable modules. Avoid putting the entire
project in one notebook.

## 6. Data architecture

### Data zones

  --------------------------------------------------------------------------
  Zone              Path                Purpose            Policy
  ----------------- ------------------- ------------------ -----------------
  Raw               `data/raw/`         Original           Immutable;
                                        downloaded source  normally
                                                           gitignored

  Interim           `data/interim/`     Temporary          Re-creatable;
                                        normalized data    gitignored

  Processed         `data/processed/`   Clean              Re-creatable;
                                        analytical/model   gitignored by
                                        data               default

  Reports           `reports/`          Aggregate tables,  Review for
                                        plots, summaries   privacy before
                                                           commit

  Models            `models/`           Serialized fitted  Usually
                                        pipelines          gitignored or
                                                           release artifact
  --------------------------------------------------------------------------

### Data flow

1.  Read source file from configured path.
2.  Record filename, SHA-256, row count, retrieval date, and source
    metadata.
3.  Validate required columns and expected grain.
4.  Profile missingness, duplicates, category values, and numeric
    ranges.
5.  Apply documented cleaning rules.
6.  Save processed data separately from raw.
7.  Build analysis tables and features from processed data.
8.  Save aggregate outputs and model artifacts with run/config metadata.

### Quality checks

-   File exists and is readable.
-   Required columns exist.
-   Expected row grain is verified.
-   Customer ID is non-null and unique if uniqueness is expected.
-   Target labels map to approved classes.
-   Tenure is numeric and non-negative.
-   Monetary values parse safely.
-   Missingness is summarized.
-   Impossible/unexpected values are flagged.
-   Input rows reconcile to retained plus excluded rows.

Do not silently drop malformed rows. Produce a validation report and
record exclusions.

## 7. Metric and analytical architecture

### Canonical metric layer

All dashboard and report metrics call the same functions. Example
interface:

``` python
def churn_prevalence(df, target_col="Churn"):
    # Return counts and the prevalence; denominator is labeled rows only.
    labeled = df[target_col].dropna()
    churned = labeled.eq("Yes").sum()
    total = len(labeled)
    return {
        "churned_count": int(churned),
        "labeled_count": int(total),
        "prevalence": churned / total if total else None,
    }
```

For a snapshot:

\[ `\text{Churn prevalence}`{=tex} =
`\frac{\text{customers labeled churned}}`{=tex}
{`\text{customers with a valid churn label}`{=tex}} \]

This is not a monthly churn rate unless the data defines an opening
population and a monthly observation period.

### Segment analysis

For each segment report its value, total labeled customers, churned
customers, prevalence, retained customers, and uncertainty interval
where appropriate. Flag very small groups. Compare both rate and volume.

### Lifecycle and cohorts

**V1:** derive documented tenure bands, for example 0--6, 7--12, 13--24,
25--48, and 49+ months. Choose boundaries based on distribution and
context; ensure no overlaps or gaps.

**True cohorts** require: - customer signup/cohort assignment date, -
dated active status or subscription events, - consistent observation
cutoff, - definition of active/retained at each cohort age.

For signup cohort (c) and age month (m):

\[ `\text{Retention}`{=tex}\_{c,m} =
`\frac{\text{customers from cohort }c\text{ active at age }m}`{=tex}
{`\text{customers originally in cohort }`{=tex}c} \]

Do not derive real signup-month cohorts from tenure alone. Synthetic
dates must be labeled simulated and must not support claims about real
seasonality.

### Lifetime / survival analysis

Use tenure summaries for V1. Kaplan--Meier analysis is appropriate only
if duration has a meaningful origin, churn event status is known,
non-churned customers are correctly right-censored at the observation
date, and the sampling/observation process is understood. Otherwise, do
not present survival estimates as reliable customer lifetime.

## 8. Machine-learning architecture

### Pipeline

``` mermaid
flowchart TD
    A[Clean customer table] --> B[Feature and target selection]
    B --> C[Leakage and ID audit]
    C --> D[Stratified train/test split]
    D --> E[Training folds]
    E --> F[Preprocessing pipeline]
    F --> G[Baseline and candidate estimators]
    G --> H[Cross-validation and model selection]
    H --> I[Final fit on training data]
    I --> J[Held-out test evaluation]
    J --> K[Threshold and calibration review]
    K --> L[Model card and saved pipeline]
```

### Preprocessing

Use scikit-learn `ColumnTransformer`:

-   Numeric: median imputation; scaling for linear models where needed.
-   Categorical: most-frequent imputation; one-hot encoding with
    unknown-category handling.
-   Estimators: logistic regression and a tree-based comparison.

Fit preprocessing only inside the pipeline on training folds. Do not
encode or scale the full dataset before splitting.

### Model comparison

1.  `DummyClassifier` baseline.
2.  `LogisticRegression` interpretable baseline.
3.  `RandomForestClassifier` or `HistGradientBoostingClassifier`
    comparison.

Choose based on validation performance, calibration, operational
constraints, and interpretability---not complexity alone.

### Evaluation

Report precision, recall, F1, ROC-AUC, average precision / PR-AUC,
confusion matrix, and calibration/Brier score where probabilities are
used. Accuracy must not be the only metric.

### Threshold selection

A 0.50 threshold is not automatically appropriate. Consider
retention-team capacity, false-positive contact cost, missed-churn cost,
and credible intervention economics. If costs are unknown, present
threshold/capacity curves and label the selected threshold as a
configurable assumption.

### Explainability

Use permutation importance on held-out data or optional SHAP.
Explanations indicate model associations, not causal churn drivers.
Investigate surprising findings and correlated predictors.

### Model artifact

Persist the complete preprocessing-plus-estimator pipeline using joblib.
Save feature list, target/class mapping, split seed, dataset
fingerprint, package versions, metrics, threshold, training timestamp,
and limitations. Never load serialized models from untrusted sources
blindly.

## 9. Dashboard architecture

Streamlit is the primary V1 interface. Load precomputed outputs rather
than retraining on every page refresh.

  -----------------------------------------------------------------------
  Page                                Contents
  ----------------------------------- -----------------------------------
  Overview                            Headline metrics, churn prevalence,
                                      counts, charge-exposure proxy

  Segments                            Contract, tenure, service, payment
                                      comparisons

  Lifecycle                           Tenure distribution and lifecycle
                                      patterns

  Model                               Comparison, confusion matrix,
                                      PR/ROC, calibration, importance

  Action plan                         Evidence, intervention hypothesis,
                                      owner, KPI, experiment
  -----------------------------------------------------------------------

Filters must update numerator and denominator. Display source and as-of
date. Show sample size beside rates. Mark unsupported cohort charts
unavailable rather than fabricating them. Avoid direct identifiers in
public demos.

## 10. Configuration

Use `configs/config.yaml` rather than hardcoding paths and column names.

``` yaml
project:
  name: customer-retention-churn
  random_seed: 42

data:
  raw_path: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  processed_path: data/processed/customers_clean.parquet
  id_column: customerID
  target_column: Churn

analysis:
  tenure_column: tenure
  monthly_charge_column: MonthlyCharges
  tenure_bins: [0, 6, 12, 24, 48, 1000]

model:
  test_size: 0.20
  cv_folds: 5
  positive_label: Yes
  threshold: 0.50
  candidates:
    - logistic_regression
    - random_forest

outputs:
  reports_dir: reports
  figures_dir: reports/figures
  models_dir: models
```

Validate configuration at startup. The threshold is a default, not an
approved optimum.

## 11. Python implementation conventions

-   Use type hints for public functions.
-   Keep functions small and single-purpose.
-   Document metric definitions and assumptions in docstrings.
-   Avoid notebook-only business logic.
-   Use `pathlib.Path`.
-   Use logging in reusable modules.
-   Set random seeds centrally.
-   Separate calculations from plotting.

Raise clear errors for missing files/columns, invalid target values,
duplicate keys where uniqueness is required, unsupported cohort
requests, missing model artifacts, and incompatible feature schemas. Do
not catch broad exceptions and continue with misleading partial results.

## 12. Testing architecture

Use pytest and small handcrafted DataFrames so unit tests do not require
downloading the full dataset.

  Test area      Example
  -------------- ---------------------------------------------------
  Validation     Missing target column raises a clear error
  Validation     Duplicate IDs are detected
  Cleaning       Blank numeric string follows documented policy
  Metrics        Prevalence uses labeled rows only
  Metrics        Empty input has defined behavior
  Segmentation   Counts and denominators reconcile
  Cohorts        Tenure bins cover valid values without overlap
  Features       ID and target are not predictors
  Modeling       Pipeline fits and predicts
  Leakage        Preprocessing is inside cross-validation pipeline
  Reporting      Output directories/files are created
  Dashboard      Data-loading helpers work independently of UI

## 13. Run metadata and observability

For each run record timestamp, source filename and SHA-256, row counts,
validation warnings, cleaning counts, config snapshot, random seed,
package versions, model parameters, and evaluation metrics.

A JSON run manifest is sufficient for a static educational dataset.
Production requires centralized logs, access controls, and monitoring.

## 14. Security and privacy

-   Keep `.env`, credentials, raw data, and row-level scores out of Git.
-   Add raw/interim/processed data, models, and local environments to
    `.gitignore`.
-   Verify dataset redistribution terms.
-   Use aggregate charts in public screenshots.
-   Avoid unnecessary sensitive features.
-   Limit access to customer-level risk outputs.
-   Do not automate customer treatment based only on model output.

## 15. Failure modes and recovery

  -----------------------------------------------------------------------
  Failure                             Expected behavior
  ----------------------------------- -----------------------------------
  CSV missing                         Stop with expected path and
                                      download guidance

  Schema mismatch                     Produce validation report and stop
                                      downstream tasks

  Invalid numeric conversion          Report affected count and apply
                                      documented policy

  No positive class in split          Fail evaluation with explanation

  Model artifact missing              Dashboard shows setup guidance

  Unsupported cohort request          Explain that event dates are
                                      required

  Empty filtered segment              Display a no-matching-customers
                                      message

  Small segment                       Flag low sample size and avoid
                                      strong conclusions
  -----------------------------------------------------------------------

## 16. Future production evolution

If adapted to a real subscription business:

1.  Replace static CSV with governed warehouse tables or approved APIs.
2.  Define event-time semantics and point-in-time feature snapshots.
3.  Use time-based train/validation/test splits.
4.  Add scheduled scoring and monitored model versions.
5.  Monitor data drift, model drift, and calibration.
6.  Evaluate campaigns with randomized experiments or robust causal
    designs.
7.  Add role-based access control and audited customer-level access.
8.  Establish retention/deletion policies and privacy review.

Do not claim these production capabilities in V1 unless implemented.

## 17. Architecture decision records

### ADR-001: Python-first implementation

**Decision:** Python for ingestion, cleaning, metrics, modeling,
visualizations, and dashboard.\
**Reason:** One language reduces integration overhead and supports
reproducibility.\
**Trade-off:** Power BI may offer stronger enterprise reporting
conventions, but is optional for V1.

### ADR-002: Streamlit dashboard

**Decision:** Streamlit for V1.\
**Reason:** Fast, Python-native interactive presentation with minimal
infrastructure.\
**Trade-off:** Less frontend flexibility than a custom web application;
adequate for this portfolio project.

### ADR-003: Snapshot before event-level cohort data

**Decision:** Begin with a verified customer snapshot and analyze tenure
bands.\
**Reason:** Available data supports segmentation and classification.\
**Trade-off:** Calendar cohorts require dated subscription events and
belong in V2.

### ADR-004: Human-in-the-loop risk prioritization

**Decision:** Predictions are decision support only.\
**Reason:** A retrospective model does not prove intervention
effectiveness and may contain bias.\
**Trade-off:** Human review and experiment measurement remain necessary.

**Maintainer:** `<your name>`\
**Last reviewed:** `<YYYY-MM-DD>`\
**Status:** Proposed architecture; revise to match implementation.
