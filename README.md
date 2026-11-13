# Customer Retention & Churn Intelligence

> An end-to-end Python analytics application that transforms
> subscription customer data into churn insights, risk estimates, and
> measurable retention actions.

## Overview

**Customer Retention & Churn Intelligence** is a data science and
analytics project designed to help subscription-based businesses
understand why customers leave, identify customer groups with elevated
churn, estimate churn risk, and prioritize evidence-based retention
actions.

The project brings together data validation, exploratory analysis,
retention metrics, customer segmentation, machine learning, model
evaluation, and an interactive Streamlit dashboard in one reproducible
workflow.

It is designed as a **decision-support system**, not simply a
churn-prediction model. A prediction is useful only when the business
can understand what it means, decide how to respond, and measure whether
the response improves retention.

### What the system does

1.  Loads and validates customer subscription records.
2.  Cleans and prepares the data for analysis.
3.  Calculates churn, retention, tenure, and recurring-charge exposure
    metrics.
4.  Identifies customer segments associated with higher churn.
5.  Examines customer lifecycle patterns using available tenure
    information.
6.  Trains and evaluates machine-learning models to estimate churn risk.
7.  Produces customer and segment insights that can support retention
    planning.
8.  Presents key findings through an interactive dashboard.
9.  Exports reports, charts, tables, and model evaluation results.

## Business problem

Customer churn represents the loss of customers over a defined period or
observation window. For subscription businesses, losing customers can
reduce recurring revenue, customer lifetime value, and the return on
customer acquisition spending.

Businesses need to understand:

-   What proportion of the observed customer base has churned?
-   Which contract, service, payment, tenure, or customer segments show
    higher churn?
-   At what stage of the customer lifecycle is churn concentrated?
-   What recurring-charge exposure is associated with churned customers?
-   Which customers should a retention team investigate first?
-   Which retention actions should be tested, and how should their
    impact be measured?

The primary objective is to help a business **reduce avoidable churn
through better measurement, risk prioritization, and controlled
retention experiments**.

## How the system works

The application follows an end-to-end analytics pipeline:

``` text
                 CUSTOMER DATASET
                        |
                        v
              DATA INGESTION
                        |
                        v
             DATA VALIDATION
       Schema, missing values, duplicates,
          invalid values and data types
                        |
                        v
                DATA CLEANING
                        |
                        v
             EXPLORATORY ANALYSIS
        Distributions, churn patterns,
          service and customer segments
                        |
                        v
           RETENTION & LIFECYCLE
        Tenure bands, retention metrics,
             segment comparisons
                        |
                        v
             FEATURE ENGINEERING
         Numerical and categorical data
                        |
                        v
             MACHINE LEARNING
        Baseline -> Logistic Regression
                  -> Random Forest
                        |
                        v
             MODEL EVALUATION
       Precision, Recall, F1, ROC-AUC,
            PR-AUC and calibration
                        |
                        v
             CUSTOMER RISK OUTPUT
        Risk estimates and prioritization
                        |
                        v
             BUSINESS RECOMMENDATIONS
         Evidence -> Action -> KPI
                        |
                        v
              STREAMLIT DASHBOARD
```

### 1. Data ingestion

The project uses the **IBM Telco Customer Churn** dataset as its
recommended starting dataset.

-   Dataset: [IBM Telco Customer Churn on
    Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
-   Common filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer-level subscription information, including
tenure, services, contract type, payment method, monthly charges, total
charges, and a churn label.

The raw dataset is kept separate from processed data so that the
original source remains unchanged and the analysis can be reproduced.

### 2. Data validation and cleaning

Before analysis, the system checks the quality and consistency of the
input data.

Typical checks include:

-   Required columns and expected data types.
-   Missing and blank values.
-   Duplicate customer identifiers.
-   Unexpected category values.
-   Invalid numerical ranges.
-   Numeric fields stored as strings.
-   Consistency of the expected one-row-per-customer data structure.

Fields such as `TotalCharges` are converted safely into numeric values.
Cleaning decisions should be documented so that their effects are
traceable.

### 3. Exploratory data analysis

The cleaned data is examined to understand the overall customer
population and identify patterns associated with churn.

The analysis includes:

-   Overall observed churn prevalence.
-   Churn by contract type.
-   Churn by tenure group.
-   Churn by internet service.
-   Churn by payment method.
-   Churn by add-on service adoption.
-   Monthly-charge distributions by churn status.
-   Customer and service segment sizes.

Charts should show the relevant sample sizes and clearly identify their
denominators. This prevents small customer groups from being mistaken
for reliable business-wide trends.

### 4. Retention and lifecycle analysis

Customers are grouped by their recorded tenure to examine how churn
prevalence varies across the customer lifecycle.

Example tenure bands:

  Tenure band     Interpretation
  --------------- -------------------------
  0--6 months     Early lifecycle
  7--12 months    Developing relationship
  13--24 months   Established customer
  25+ months      Longer-tenure customer

The standard Telco snapshot generally contains tenure but not exact
signup dates, cancellation dates, or dated monthly activity records.

Therefore, this version supports **tenure-band/lifecycle analysis**, not
genuine signup-month cohort analysis or month-by-month retention curves.
True cohort analysis requires dated signup and activity records. Dates
must never be fabricated and presented as observed data.

### 5. Feature engineering

Customer information is prepared for machine learning.

-   Numerical features are identified and processed.
-   Categorical features are encoded.
-   Missing values are handled using training-fitted preprocessing.
-   The customer identifier is excluded from predictive features.
-   Preprocessing is incorporated into a scikit-learn pipeline to
    prevent information from the test set leaking into training.

The target variable is `Churn`.

### 6. Churn prediction

The modeling workflow starts with a simple baseline and compares it with
more capable models.

  -----------------------------------------------------------------------
  Model                               Purpose
  ----------------------------------- -----------------------------------
  Dummy Classifier                    Establishes a baseline for
                                      comparison

  Logistic Regression                 Provides an interpretable
                                      classification baseline

  Random Forest                       Captures more complex feature
                                      relationships
  -----------------------------------------------------------------------

The workflow uses stratified train/test splitting and cross-validation
on the training data. The test set is reserved for final evaluation.

The model estimates the likelihood that a customer belongs to the churn
class, based on the available customer attributes.

These estimates indicate **risk or association**, not proof that a
particular customer will churn or that a particular intervention will
prevent churn.

### 7. Model evaluation

Accuracy alone is not sufficient for evaluating a churn model,
especially when churn and non-churn classes are imbalanced.

The evaluation includes:

-   Confusion matrix.
-   Precision.
-   Recall.
-   F1-score.
-   ROC-AUC.
-   PR-AUC.
-   Probability calibration, where appropriate.

The classification threshold should be selected based on business costs,
the relative importance of missed churners and false alerts, and the
retention team's capacity. A threshold of 0.5 should not be assumed to
be optimal.

### 8. Retention decision support

The predictive output is translated into a review and prioritization
process.

Customer prioritization can consider:

-   Estimated churn risk.
-   Monthly recurring-charge exposure proxy.
-   Customer eligibility for an intervention.
-   Intervention cost.
-   Available evidence about treatment effectiveness.
-   Retention team capacity.

The resulting insights can help teams decide which segments to
investigate and which retention strategies to test.

A high churn-risk score does not establish that a discount, support
call, or service change will prevent churn. The effectiveness of
interventions should be measured through controlled experiments.

## Technology stack

  Technology     Role
  -------------- -------------------------------------------------
  Python 3.11+   Main runtime and application logic
  pandas         Data loading, cleaning, and transformation
  NumPy          Numerical operations
  SciPy          Statistical analysis
  statsmodels    Statistical tests and analysis
  Matplotlib     Visualizations
  Seaborn        Statistical charts
  Plotly         Interactive visualizations
  scikit-learn   Preprocessing, machine learning, and evaluation
  Streamlit      Interactive analytics dashboard
  PyYAML         Configuration management
  joblib         Model serialization
  pytest         Automated testing
  DuckDB         Optional SQL-based analysis

## Project structure

The following is the intended repository organization. Keep the actual
filenames and entry points synchronized with the implementation.

``` text
customer-retention-churn/
│
├── README.md
├── PRD.md
├── ARCHITECTURE.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_quality_eda.ipynb
│   ├── 02_retention_segmentation.ipynb
│   └── 03_model_experiments.ipynb
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── executive_summary.md
│
├── models/
│
├── src/
│   └── retention/
│       ├── config.py
│       ├── data/
│       │   ├── load.py
│       │   ├── validate.py
│       │   └── clean.py
│       ├── analysis/
│       │   ├── metrics.py
│       │   ├── cohorts.py
│       │   ├── segmentation.py
│       │   └── lifetime.py
│       ├── features/
│       │   └── build.py
│       ├── modeling/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── explain.py
│       ├── reporting/
│       │   └── charts.py
│       └── app/
│           └── streamlit_app.py
│
└── tests/
    ├── test_data_validation.py
    ├── test_metrics.py
    ├── test_cohorts.py
    └── test_features.py
```

## Getting started

### Prerequisites

Install the following before running the project:

-   Python 3.11 or newer.
-   Git.
-   A local copy of the customer churn dataset.
-   VS Code or JupyterLab (optional).

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/customer-retention-churn.git
cd customer-retention-churn
```

### 2. Create a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A starting `requirements.txt` can contain:

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

For a reproducible release, pin the package versions that have actually
been tested.

### 4. Add the dataset

Download the IBM Telco Customer Churn dataset and place the CSV in:

``` text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Do not modify the raw source file.

Record the dataset source, retrieval date, license, checksum, and any
relevant transformations in `data/README.md`.

## Running the project

Run commands from the repository root with the virtual environment
activated.

The following commands represent the intended module-based execution
interface. They require the corresponding Python modules and
command-line entry points to exist in the repository.

### Step 1: Validate the dataset

``` bash
python -m retention.data.validate --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

This step checks the source data before it enters the analysis pipeline.

### Step 2: Clean and prepare the data

``` bash
python -m retention.data.clean
```

This step prepares the analysis-ready dataset while preserving the
original raw file.

### Step 3: Run descriptive analytics

``` bash
python -m retention.analysis.metrics
python -m retention.analysis.segmentation
python -m retention.analysis.lifetime
```

These steps generate churn metrics, customer segment comparisons, and
lifecycle insights.

### Step 4: Train the churn model

``` bash
python -m retention.modeling.train
```

The training workflow prepares features, fits the configured models, and
saves the resulting model artifacts if implemented by the training
module.

### Step 5: Evaluate the model

``` bash
python -m retention.modeling.evaluate
```

Review the classification metrics and model evaluation outputs before
using risk estimates for prioritization.

### Step 6: Launch the dashboard

``` bash
streamlit run src/retention/app/streamlit_app.py
```

Streamlit starts a local web application. Open the local URL printed in
the terminal to use the dashboard.

### Step 7: Run automated tests

``` bash
pytest -q
```

Run the tests after installation and whenever the data-processing,
metric, feature-engineering, or modeling logic changes.

> **Important:** The commands above are valid run instructions only when
> the repository implements the corresponding modules. If the
> implementation uses different filenames, package paths, or CLI
> commands, update this section to match the actual code. Do not publish
> placeholder commands as verified execution steps.

## Dashboard

The Streamlit dashboard is organized around five business-facing areas.

### 1. Executive overview

Provides a high-level view of:

-   Total customer count.
-   Observed churn prevalence.
-   Retention complement.
-   Recurring-charge exposure proxy.
-   Key customer segments.

### 2. Churn drivers

Allows users to examine churn patterns across:

-   Contract types.
-   Tenure bands.
-   Internet and other services.
-   Payment methods.
-   Monthly-charge bands.

### 3. Lifecycle analysis

Displays tenure-band comparisons and lifecycle patterns. A true cohort
heatmap should only be displayed when valid dated signup and activity
data is available.

### 4. Customer risk

Presents the distribution of estimated churn risk, configurable
thresholds, and a privacy-conscious review queue for customers who may
warrant investigation.

### 5. Recommendations

Connects observed evidence to proposed business actions. Each
recommendation should specify:

  Field               Purpose
  ------------------- -----------------------------------------
  Segment             Customers the finding concerns
  Evidence            Observed counts, rates, and sample size
  Proposed action     Retention intervention to test
  Owner               Responsible team or role
  Success metric      How the intervention will be evaluated
  Experiment design   How impact will be measured

Dashboard filters must update numerators and denominators consistently.
Metric definitions and the data-as-of date should be visible.

## Metrics and interpretation

The project distinguishes customer-level churn from revenue-level churn.

  -----------------------------------------------------------------------
  Metric                  Definition              Important qualification
  ----------------------- ----------------------- -----------------------
  Observed dataset churn  Customers labeled       Snapshot-level
  prevalence              `Churn = Yes` divided   proportion, not
                          by all labeled          automatically monthly
                          customers               churn

  Segment churn           Churned customers in a  Always show the segment
  prevalence              segment divided by      size
                          labeled customers in    
                          that segment            

  Retention complement    1 − churn prevalence    Must use the same
                                                  population and label
                                                  definition

  MRR                     Sum of active monthly   Requires verified
                          recurring subscription  recurring-charge
                          charges                 semantics and active
                                                  subscription records

  Revenue churn           Opening recurring       Requires dated period
                          revenue lost divided by data
                          opening recurring       
                          revenue                 

  ARPU                    Revenue in a period     State the period and
                          divided by average      revenue basis
                          active customers        

  Customer lifetime       Observed active         Snapshot tenure is not
                          duration or a clearly   automatically a full
                          stated estimate         survival estimate

  CLV/LTV                 Expected customer value Requires margin,
                          over a defined horizon  discount, retention,
                                                  and revenue assumptions
  -----------------------------------------------------------------------

For the static Telco snapshot, the fraction of `Churn = Yes` is reported
as **observed dataset churn prevalence**.

`MonthlyCharges` may be used as a rough recurring-charge exposure proxy
only when its assumptions are stated. `TotalCharges` is cumulative and
must not be treated as MRR.

## Expected outputs

The analysis workflow is designed to produce a set of reusable reports
and artifacts.

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

These are expected artifact names; the actual files are produced only
when their corresponding reporting and modeling steps are implemented
and executed.

Do not publish identifiable customer-level predictions or raw personal
data in a public repository.

## Responsible use

This project is intended for educational, analytical, and
decision-support use with data that the operator is permitted to
process.

-   Remove direct identifiers from model features.
-   Restrict access to customer-level risk scores.
-   Evaluate subgroup performance where appropriate and lawful.
-   Do not automatically deny service, raise prices, or impose penalties
    based on a risk score.
-   Use predictions to identify opportunities for helpful customer
    support, not to manipulate customers.
-   Do not claim that a predictive association proves causation.
-   Validate retention interventions through controlled experiments.
-   Do not publish restricted raw data or identifiable customer records.

This sample-data project is not, by itself, a live production
customer-retention system.

## Limitations

1.  **Static snapshot:** The standard Telco dataset does not provide a
    dated sequence of customer activity, so it cannot establish
    month-by-month churn trends.
2.  **No genuine signup cohorts:** Tenure bands are supported, but true
    signup-month cohorts require actual signup dates and dated activity
    events.
3.  **Revenue assumptions:** Monthly charges are only a proxy for
    recurring-charge exposure unless active subscription and billing
    semantics are verified.
4.  **Prediction versus causation:** Model scores estimate risk; they do
    not prove why a customer will churn or which action will stop it.
5.  **CLV assumptions:** A defensible lifetime-value estimate requires
    additional assumptions and data, including margin, horizon, discount
    rate, and retention.
6.  **Dataset provenance:** Verify the exact dataset source, license,
    schema, and retrieval date before use.

## Project quality checklist

Before publishing a release, verify that:

-   [ ] Dataset source, license, and retrieval date are documented.
-   [ ] Raw data is preserved and excluded from Git where required.
-   [ ] Schema, missingness, duplicates, and invalid values are checked.
-   [ ] Cleaning decisions are documented.
-   [ ] Metric definitions and denominators are documented.
-   [ ] Snapshot and cohort limitations are disclosed.
-   [ ] No fabricated dates or unsupported causal claims are used.
-   [ ] Train/test separation and leakage controls are implemented.
-   [ ] Candidate models are compared with a baseline.
-   [ ] Appropriate metrics for class imbalance are reported.
-   [ ] The classification threshold is justified or clearly
    configurable.
-   [ ] Actionable findings are supported by counts, rates, and sample
    sizes.
-   [ ] Dashboard filters and metric definitions are verified.
-   [ ] Automated tests pass from a clean environment.
-   [ ] Public outputs contain no restricted raw data or identifiable
    predictions.
-   [ ] The final report separates observed facts, model estimates, and
    recommendations.

## References

-   Future Interns Task 2 (2026): Customer Retention & Churn Analysis.
-   [IBM Telco Customer Churn dataset ---
    Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
-   [scikit-learn documentation](https://scikit-learn.org/stable/)
-   [pandas documentation](https://pandas.pydata.org/docs/)
-   [Streamlit documentation](https://docs.streamlit.io/)

------------------------------------------------------------------------

**Project:** Customer Retention & Churn Intelligence\
**Category:** Data Science & Analytics\
**Primary language:** Python\
**Application interface:** Streamlit\
**Dataset:** IBM Telco Customer Churn (subject to source and license
verification)
