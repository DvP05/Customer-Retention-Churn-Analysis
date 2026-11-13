# Customer Retention & Churn Intelligence

> **Future Interns --- Data Science & Analytics, Task 2 (2026)**\
> A Python-first analytics project that turns subscription data into
> measurable retention decisions.

## 1. Project overview

Customer churn is the loss of customers over a defined period. In
subscription businesses, churn affects recurring revenue, customer
lifetime value, acquisition efficiency, and growth. This project creates
an end-to-end workflow to:

1.  Validate and clean customer/subscription records.
2.  Measure churn, retention, customer lifetime, and revenue exposure.
3.  Identify customer segments associated with elevated churn.
4.  Analyze lifecycle/cohort patterns where the data supports them.
5.  Train and evaluate a machine-learning model that estimates churn
    risk.
6.  Translate findings into practical, measurable retention actions.
7.  Publish reproducible results, visualizations, and an executive
    dashboard.

This is a **decision-support project**, not merely a classifier. A model
score is useful only when it is evaluated against an appropriate
baseline and connected to an intervention that the business can test.

## 2. Business problem and objectives

A subscription business needs to answer:

-   What proportion of customers churn, and how is it changing?
-   Which contract, service, payment, tenure, or customer segments have
    higher churn?
-   At what point in the customer lifecycle is churn concentrated?
-   How much recurring-charge exposure is associated with churn?
-   Which customers should a retention team investigate first?
-   Which retention interventions should be tested, and how will impact
    be measured?

**Primary objective:** reduce avoidable churn by identifying high-risk
segments and enabling targeted, measurable retention experiments.

**Analytical objectives:** establish trustworthy baseline metrics;
quantify associations; build a leakage-safe predictive model; prioritize
customers using risk, value, and eligibility; and provide reproducible
evidence for recommendations.

## 3. Dataset and analytical boundaries

### Recommended starting dataset

Use the **IBM Telco Customer Churn** dataset, commonly distributed
through Kaggle:

-   Kaggle:
    <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
-   Common filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Verify the exact source, license, file schema, and provenance before
use. Dataset mirrors may differ.

### Critical limitation: cohort analysis

The standard Telco dataset generally contains `tenure`, but not exact
signup dates, cancellation dates, or a dated sequence of monthly
activity. Therefore:

-   It supports **lifecycle/tenure-band analysis** (for example, 0--6,
    7--12, 13--24 months).
-   It does **not** support true signup-month cohorts or month-by-month
    retention curves without additional dated records.
-   Do not fabricate signup dates and present them as observed facts.
-   For true cohort analysis, obtain a dataset with signup dates and
    dated subscription/activity events. Synthetic dates must be
    explicitly labeled synthetic.

### Common data dictionary

  -----------------------------------------------------------------------
  Field                   Typical meaning         Notes
  ----------------------- ----------------------- -----------------------
  `customerID`            Customer identifier     Never use as a model
                                                  predictor

  `gender`                Recorded gender         Use only where
                          category                justified

  `SeniorCitizen`         Senior-citizen          Often 0/1
                          indicator               

  `Partner`, `Dependents` Household attributes    Categorical

  `tenure`                Months with company     Lifecycle measure

  `PhoneService`,         Phone services          Categorical
  `MultipleLines`                                 

  `InternetService`       Internet service type   Categorical

  `OnlineSecurity`,       Add-on services         Categorical
  `OnlineBackup`,                                 
  `DeviceProtection`                              

  `TechSupport`,          Add-on services         Categorical
  `StreamingTV`,                                  
  `StreamingMovies`                               

  `Contract`              Contract type           Categorical

  `PaperlessBilling`      Billing preference      Categorical

  `PaymentMethod`         Payment method          Categorical

  `MonthlyCharges`        Monthly charge          Potential
                                                  recurring-charge proxy

  `TotalCharges`          Accumulated charges     Often string-like in
                                                  raw CSV

  `Churn`                 Whether customer        Target label
                          churned                 
  -----------------------------------------------------------------------

Confirm every field against the downloaded data.

## 4. Metric definitions and policy

Document the observation window and denominator for every metric. Do not
mix customer-level and revenue-level churn.

  -----------------------------------------------------------------------
  Metric                  Definition              Interpretation
  ----------------------- ----------------------- -----------------------
  Dataset churn           Churned labeled         Snapshot-level
  prevalence              customers / all labeled fraction, not
                          customers               automatically monthly
                                                  churn

  Segment churn           Churned customers in    Include group size
  prevalence              segment / labeled       
                          customers in segment    

  Retention complement    1 − churn prevalence    Only for the same
                                                  population and label
                                                  definition

  MRR                     Sum of active monthly   Requires verified
                          recurring subscription  recurring-charge
                          charges                 semantics

  Revenue churn           Opening recurring       Requires dated period
                          revenue lost / opening  data
                          recurring revenue       

  ARPU                    Revenue in period /     State period and
                          average active          revenue basis
                          customers               

  Customer lifetime       Observed active         Do not confuse snapshot
                          duration or a clearly   tenure with a full
                          stated estimate         survival estimate

  CLV/LTV                 Expected customer value Requires margin,
                          over a defined horizon  discount, and retention
                                                  assumptions
  -----------------------------------------------------------------------

For the static Telco snapshot, label the fraction of `Churn = Yes` as
**observed dataset churn prevalence**, not monthly churn.
`MonthlyCharges` can be a rough recurring-charge exposure proxy only
after explaining the assumption. `TotalCharges` is cumulative, not MRR.

## 5. Technology stack

  -----------------------------------------------------------------------
  Layer                   Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Runtime                 Python 3.11+            Main implementation

  Data                    pandas, NumPy           Cleaning and
                                                  transformation

  Validation              pandera or explicit     Schema and quality
                          pandas checks           controls

  Statistics              SciPy, statsmodels      Tests and intervals

  Visualization           Matplotlib, Seaborn,    Charts
                          Plotly                  

  ML                      scikit-learn            Pipelines, models,
                                                  metrics

  Explainability          Permutation importance; Model interpretation
                          SHAP optional           

  Dashboard               Streamlit               Python-native
                                                  interactive UI

  SQL extension           DuckDB                  Optional SQL analysis

  Testing                 pytest                  Automated tests

  Version control         Git/GitHub              Portfolio publishing
  -----------------------------------------------------------------------

Power BI is optional. Streamlit keeps the V1 application predominantly
Python.

## 6. Recommended repository structure

``` text
customer-retention-churn/
├── README.md
├── PRD.md
├── ARCHITECTURE.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/config.yaml
├── data/
│   ├── raw/                 # original data; gitignored
│   ├── interim/             # intermediate data; gitignored
│   ├── processed/           # cleaned data; gitignored by default
│   └── README.md            # provenance and download notes
├── notebooks/
│   ├── 01_data_quality_eda.ipynb
│   ├── 02_retention_segmentation.ipynb
│   └── 03_model_experiments.ipynb
├── reports/
│   ├── figures/
│   ├── tables/
│   └── executive_summary.md
├── models/                  # serialized models; usually gitignored
├── src/retention/
│   ├── config.py
│   ├── data/{load.py,validate.py,clean.py}
│   ├── analysis/{metrics.py,cohorts.py,segmentation.py,lifetime.py}
│   ├── features/build.py
│   ├── modeling/{train.py,evaluate.py,explain.py}
│   ├── reporting/charts.py
│   └── app/streamlit_app.py
└── tests/
    ├── test_data_validation.py
    ├── test_metrics.py
    ├── test_cohorts.py
    └── test_features.py
```

Keep reusable logic in Python modules; use notebooks for exploration
rather than as the only location of business logic.

## 7. Setup

### Prerequisites

-   Python 3.11 or newer
-   Git
-   Local copy of the selected dataset
-   Optional: VS Code or JupyterLab

### Install

``` bash
git clone https://github.com/<your-username>/customer-retention-churn.git
cd customer-retention-churn
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt`:

``` text
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
plotly
scikit-learn
streamlit
pyyaml
joblib
pytest
duckdb
```

Pin tested package versions before publishing a release.

### Add the dataset

Place the downloaded CSV at:

``` text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Keep raw data out of Git unless its license explicitly permits
redistribution. Record source URL, retrieval date, license, checksum,
and transformations in `data/README.md`.

## 8. Intended execution workflow

The following commands describe the intended CLI. Implement these entry
points or update the README to match the actual package.

``` bash
# Validate and profile the source
python -m retention.data.validate --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Clean and export analysis-ready data
python -m retention.data.clean

# Run descriptive analytics
python -m retention.analysis.metrics
python -m retention.analysis.segmentation
python -m retention.analysis.lifetime

# Train and evaluate models
python -m retention.modeling.train
python -m retention.modeling.evaluate

# Launch the dashboard
streamlit run src/retention/app/streamlit_app.py

# Run tests
pytest -q
```

Do not leave documentation commands pointing to modules that have not
been implemented.

## 9. Analysis workflow

### A. Data understanding and quality

1.  Read the source without modifying the raw copy.
2.  Inspect rows, columns, data types, duplicate IDs, missingness,
    categories, and numeric ranges.
3.  Verify one row per customer if that is the expected grain.
4.  Convert numeric-looking strings such as `TotalCharges` safely.
5.  Investigate blanks, sentinels, and inconsistent category labels.
6.  Record every cleaning decision and its effect.

### B. Exploratory analysis

Analyze overall churn prevalence; churn by contract, tenure band,
internet service, payment method, and add-on adoption; monthly charge
distribution by churn status; service combinations; segment sizes; and
feature distributions.

Every chart should state its business question, denominator, and
interpretation. Show sample sizes because high rates in tiny groups can
be unstable.

### C. Retention and lifetime

-   Create lifecycle bands from tenure.
-   Compare churn prevalence across bands.
-   Use dated signup cohorts only when actual dates and activity states
    exist.
-   Use Kaplan--Meier survival analysis only when durations, event
    indicators, and censoring are valid.
-   Do not estimate CLV without stating margin, horizon, discount rate,
    and revenue assumptions.

### D. Predictive modeling

-   Start with a `DummyClassifier` baseline and logistic regression.
-   Compare with a tree-based model such as Random Forest.
-   Use stratified splitting.
-   Fit preprocessing only on training data inside a scikit-learn
    `Pipeline` / `ColumnTransformer`.
-   Use cross-validation on training data for model selection.
-   Keep the test set untouched until final evaluation.
-   Report confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, and
    calibration where appropriate.
-   Choose the threshold based on business costs and team capacity, not
    automatically 0.5.
-   Audit for leakage and subgroup performance.

### E. Business action

Prioritize using predicted risk, recurring-charge/customer-value proxy,
eligibility, intervention cost, and credible treatment-effect evidence.
A churn model alone cannot establish that a discount, call, or feature
intervention prevents churn. Recommend controlled experiments.

## 10. Dashboard specification

Suggested Streamlit pages:

1.  **Executive overview:** customer count, churn prevalence, retention
    complement, recurring-charge exposure proxy, key segments.
2.  **Churn drivers:** churn by contract, tenure, service, payment
    method, and charge band.
3.  **Lifecycle/cohorts:** tenure-band analysis; true cohort heatmap
    only with dated events.
4.  **Customer risk:** risk distribution, threshold controls,
    privacy-safe review queue.
5.  **Recommendations:** segment, observed evidence, proposed action,
    owner, success metric, experiment design.

Filters should update numerator and denominator consistently. Display
data-as-of date and metric definitions.

## 11. Expected outputs

``` text
reports/
├── executive_summary.md
├── data_quality_report.md
├── churn_metrics.csv
├── segment_analysis.csv
├── model_metrics.json
├── figures/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_tenure.png
│   ├── tenure_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── tables/
    ├── segment_summary.csv
    └── customer_risk_scores.csv
```

A cohort heatmap belongs here only if valid dated cohort data exists.
Never publish identifiable customer-level predictions or raw personal
data in a public repository.

## 12. Acceptance checklist

-   [ ] Dataset source, license, and retrieval date documented.
-   [ ] Raw data preserved and excluded from Git where required.
-   [ ] Schema, missingness, duplicates, and invalid values checked.
-   [ ] Metric definitions and denominators documented.
-   [ ] Snapshot limitations clearly disclosed.
-   [ ] No fabricated dates or unsupported causal claims.
-   [ ] Train/test separation and leakage controls implemented.
-   [ ] Baseline compared with candidate models.
-   [ ] Metrics appropriate for class imbalance reported.
-   [ ] Threshold justified by costs/capacity or labeled as
    configurable.
-   [ ] At least three actionable findings supported by counts and
    rates.
-   [ ] Dashboard filters and definitions verified.
-   [ ] Tests pass from a clean environment.
-   [ ] Public repository contains documentation, not restricted raw
    data.
-   [ ] Final report separates observed facts, model estimates, and
    recommendations.

## 13. Responsible use

Use only data you are permitted to process. Remove direct identifiers
from model features and public outputs. Restrict access to
customer-level risk scores. Evaluate relevant subgroup performance where
appropriate and lawful. Do not automatically deny service, raise prices,
or impose penalties based on risk scores. Use predictions to offer
helpful support, not manipulate customers. State that this is an
educational/sample-data project, not a live production system.

## 14. Suggested milestones

  -----------------------------------------------------------------------
  Milestone               Work                    Exit criterion
  ----------------------- ----------------------- -----------------------
  1\. Data foundation     Source, provenance,     Reproducible clean
                          schema, quality         dataset

  2\. Descriptive         KPI definitions, EDA,   Validated metrics and
  analytics               segments                charts

  3\. Lifecycle analysis  Tenure bands / valid    Qualified retention
                          cohorts                 findings

  4\. Predictive          Baseline, models,       Leakage-safe model
  analytics               evaluation              report

  5\. Decision layer      Risk prioritization,    Actions tied to
                          recommendations         evidence and KPIs

  6\. Productization      Streamlit, tests, docs  Runs locally from
                                                  documented steps

  7\. Portfolio release   GitHub, screenshots,    Reproducible public
                          LinkedIn                project
  -----------------------------------------------------------------------

## 15. Portfolio presentation

Suggested narrative (replace with actual results):

> I built a Python-first customer retention analytics workflow that
> validates subscription data, measures churn and lifecycle patterns,
> identifies high-risk customer segments, and evaluates predictive
> models. I translated findings into a decision-support dashboard and
> testable retention recommendations. I documented dataset limitations,
> metric definitions, model validation, and responsible-use
> considerations.

Never invent accuracy, churn reduction, revenue saved, or business
impact.

## 16. References

-   Future Interns Task 2 (2026): Customer Retention & Churn Analysis.
-   IBM Telco Customer Churn dataset distribution:
    <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
-   scikit-learn: <https://scikit-learn.org/stable/>
-   pandas: <https://pandas.pydata.org/docs/>
-   Streamlit: <https://docs.streamlit.io/>

**Maintainer:** `<your name>`\
**Last updated:** `<YYYY-MM-DD>`\
**Status:** Planning / implementation
