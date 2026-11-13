"""
Customer Retention & Churn Intelligence Hub.
Interactive executive dashboard for subscription analytics, segmentation,
lifecycle analysis, predictive risk modeling, and operational action planning.
"""

from pathlib import Path
from typing import Dict, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retention.analysis.cohorts import check_cohort_capability
from retention.analysis.metrics import calculate_headline_kpis, churn_prevalence
from retention.analysis.segmentation import calculate_wilson_ci
from retention.config import config

# Page setup
st.set_page_config(
    page_title="Customer Retention & Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-subtext {
        font-size: 0.80rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .action-card {
        background: #ffffff;
        color: #334155;
        border-left: 4px solid #1e3a8a;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .action-card h4 {
        color: #0f172a;
    }
    .action-card p {
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_analytical_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    parquet_path = config.processed_parquet
    df_clean = pd.read_parquet(parquet_path)

    # Tables
    kpi_file = config.tables_dir / "churn_kpi_summary.csv"
    kpi_summary = pd.read_csv(kpi_file) if kpi_file.exists() else pd.DataFrame([calculate_headline_kpis(df_clean)])

    seg_file = config.tables_dir / "segment_analysis.csv"
    seg_summary = pd.read_csv(seg_file) if seg_file.exists() else pd.DataFrame()

    tenure_file = config.tables_dir / "tenure_band_summary.csv"
    tenure_summary = pd.read_csv(tenure_file) if tenure_file.exists() else pd.DataFrame()

    tradeoff_file = config.tables_dir / "threshold_capacity_tradeoff.csv"
    tradeoffs = pd.read_csv(tradeoff_file) if tradeoff_file.exists() else pd.DataFrame()

    metrics_file = config.reports_dir / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {}

    return df_clean, kpi_summary, seg_summary, tenure_summary, tradeoffs, model_metrics


@st.cache_data
def load_risk_queue() -> pd.DataFrame:
    queue_file = config.tables_dir / "customer_risk_queue.csv"
    if queue_file.exists():
        return pd.read_csv(queue_file)
    return pd.DataFrame()


def render_sidebar() -> str:
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("**Customer Retention & Churn Intelligence**")
    st.sidebar.caption("Future Interns — Task 2 (2026)")

    page = st.sidebar.radio(
        "Select Analytical View:",
        [
            "1. Executive Overview",
            "2. Segment Explorer",
            "3. Lifecycle & Tenure Patterns",
            "4. Predictive Model Diagnostics",
            "5. Action Plan & Review Queue",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ℹ️ Dataset Scope")
    st.sidebar.info(
        "**Source:** IBM Telco Customer Churn\n\n"
        "**Observation Scope:** Static cross-sectional customer snapshot ($N=7,043$).\n\n"
        "**Target Definition:** Retrospective churn indicator (`Churn == 'Yes'`)."
    )
    return page


def page_executive_overview(df: pd.DataFrame, kpi_df: pd.DataFrame, seg_df: pd.DataFrame) -> None:
    st.header("Executive Overview — Retention & Exposure Health")
    st.markdown(
        "A holistic diagnosis of customer attrition prevalence, recurring revenue exposure, "
        "and critical vulnerability hotspots across the customer base."
    )

    kpi = kpi_df.iloc[0].to_dict()

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tracked Accounts</div>
                <div class="metric-value">{kpi['total_customers']:,}</div>
                <div class="metric-subtext">Verified 1:1 customer grain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <div class="metric-label">Snapshot Churn Prevalence</div>
                <div class="metric-value" style="color: #ef4444;">{kpi['churn_prevalence_pct']:.2f}%</div>
                <div class="metric-subtext">{kpi['churned_customers']:,} total churned accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #10b981;">
                <div class="metric-label">Retention Complement</div>
                <div class="metric-value" style="color: #10b981;">{kpi['retention_complement_pct']:.2f}%</div>
                <div class="metric-subtext">{kpi['retained_customers']:,} retained accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #0284c7;">
                <div class="metric-label">Monthly Charge Exposure</div>
                <div class="metric-value" style="color: #0284c7;">${kpi['churn_exposure_proxy']:,.2f}</div>
                <div class="metric-subtext">{kpi['exposure_share_pct']:.1f}% of total recurring billings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row
    col_chart1, col_chart2 = st.columns([1, 1.3])

    with col_chart1:
        st.subheader("Customer Base Composition")
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Retained Customers", "Churned Customers"],
                    values=[kpi["retained_customers"], kpi["churned_customers"]],
                    hole=0.55,
                    marker=dict(colors=["#10b981", "#ef4444"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("Top Vulnerability Hotspots (by Churn Rate)")
        if not seg_df.empty:
            top_vuln = seg_df.sort_values("churn_rate_pct", ascending=True).tail(6)
            fig_bar = px.bar(
                top_vuln,
                x="churn_rate_pct",
                y="segment_value",
                orientation="h",
                text="churn_rate_pct",
                color="churn_rate_pct",
                color_continuous_scale="Reds",
                labels={"churn_rate_pct": "Churn Rate (%)", "segment_value": "Segment"},
            )
            fig_bar.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

    # Executive narrative card
    st.markdown("### 📋 Executive Diagnostic Summary")
    st.info(
        "1. **What happened:** 26.5% of analyzed customer accounts terminated service, representing $139.1K in exposed monthly billings (~30.5% of recurring charge volume).\n\n"
        "2. **Where attrition concentrates:** Churn is heavily concentrated in early tenure (0–6 months at 52.9%), Month-to-month contracts (42.7%), Electronic check payers (45.3%), and Fiber Optic accounts lacking dedicated Tech Support (41.6%).\n\n"
        "3. **What to test next:** Deploy proactive 30-day onboarding assistance for new subscribers, incentivize annual plan renewal commitments, and test payment migration credits toward automated ACH billing."
    )


def page_segment_explorer(df: pd.DataFrame, seg_df: pd.DataFrame) -> None:
    st.header("Segment Explorer & Multi-Dimensional Slicer")
    st.markdown(
        "Examine customer attrition across contract terms, service technologies, and payment preferences. "
        "Every rate is reported with sample size ($N$) and **Wilson Score 95% Confidence Intervals**."
    )

    # Interactive filters
    f1, f2, f3 = st.columns(3)
    with f1:
        contract_filter = st.multiselect("Contract Type", options=df["Contract"].unique(), default=df["Contract"].unique())
    with f2:
        internet_filter = st.multiselect("Internet Service", options=df["InternetService"].unique(), default=df["InternetService"].unique())
    with f3:
        payment_filter = st.multiselect("Payment Method", options=df["PaymentMethod"].unique(), default=df["PaymentMethod"].unique())

    # Apply filters
    filtered_df = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["PaymentMethod"].isin(payment_filter)
    ]

    filtered_n = len(filtered_df)
    filtered_churned = int((filtered_df["Churn"] == "Yes").sum())
    filtered_rate = (filtered_churned / filtered_n * 100) if filtered_n > 0 else 0.0
    ci_low, ci_high = calculate_wilson_ci(filtered_churned, filtered_n)

    if filtered_n < 30:
        st.warning(f"⚠️ **Low Sample Alert:** Current filter selection contains only {filtered_n} customers. Statistical estimates may have high variance.")

    # KPI summary of filtered slice
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Selected Segment Volume", f"{filtered_n:,}")
    s2.metric("Segment Churned Accounts", f"{filtered_churned:,}")
    s3.metric("Segment Churn Prevalence", f"{filtered_rate:.2f}%")
    s4.metric("95% Confidence Interval", f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Comparison Table
    st.subheader("Aggregated Segment Analysis Table")
    if not seg_df.empty:
        st.dataframe(
            seg_df.style.format({
                "churn_rate_pct": "{:.2f}%",
                "ci_95_lower_pct": "{:.2f}%",
                "ci_95_upper_pct": "{:.2f}%",
                "exposure_proxy": "${:,.2f}",
                "total_monthly_charges": "${:,.2f}",
                "customer_count": "{:,}",
                "churned_count": "{:,}",
            }),
            use_container_width=True,
            height=380,
        )


def page_lifecycle_patterns(df: pd.DataFrame, tenure_df: pd.DataFrame) -> None:
    st.header("Customer Lifecycle & Tenure Dynamics")
    st.markdown(
        "Analyze customer duration patterns across operational tenure bands. "
        "Identifies critical inflection points where customer drop-off is concentrated."
    )

    # Prominent callout on cohort limitations
    _, notice = check_cohort_capability(df)
    st.warning(f"📌 **Analytical Boundary Notice:** {notice}")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Tenure Distribution by Churn Status")
        fig_tenure = px.histogram(
            df,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=36,
            color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
            labels={"tenure": "Tenure (Months)", "count": "Customer Count"},
            opacity=0.75,
        )
        fig_tenure.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            legend=dict(title="Churn Status", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with c2:
        st.subheader("Tenure Band Churn Prevalence")
        if not tenure_df.empty:
            fig_band = px.bar(
                tenure_df,
                x="tenure_band",
                y="churn_prevalence_pct",
                text="churn_prevalence_pct",
                color="churn_prevalence_pct",
                color_continuous_scale="Blues",
                labels={"churn_prevalence_pct": "Churn Rate (%)", "tenure_band": "Lifecycle Band"},
            )
            fig_band.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=340,
            )
            fig_band.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_band, use_container_width=True)

    st.subheader("Lifecycle Tenure Band Summary Table")
    if not tenure_df.empty:
        st.dataframe(
            tenure_df.style.format({
                "share_of_total_customers_pct": "{:.2f}%",
                "churn_prevalence_pct": "{:.2f}%",
                "retention_rate_pct": "{:.2f}%",
                "mean_monthly_charges": "${:,.2f}",
                "exposure_proxy": "${:,.2f}",
                "customer_count": "{:,}",
                "churned_count": "{:,}",
                "retained_count": "{:,}",
            }),
            use_container_width=True,
        )


def page_model_evaluation(tradeoffs: pd.DataFrame, model_metrics: Dict) -> None:
    st.header("Predictive Model Diagnostics & Capacity Optimizer")
    st.markdown(
        "Evaluate model performance against baseline, examine precision-recall calibration, "
        "and optimize the decision threshold based on customer review capacity."
    )

    if model_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC Score", f"{model_metrics.get('roc_auc', 0):.4f}")
        m2.metric("PR-AUC (Avg Precision)", f"{model_metrics.get('pr_auc', 0):.4f}")
        m3.metric("Recall (Sensitivity)", f"{model_metrics.get('recall', 0):.4f}")
        m4.metric("Precision", f"{model_metrics.get('precision', 0):.4f}")
        m5.metric("Brier Calibration Score", f"{model_metrics.get('brier_score', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Threshold Simulator
    st.subheader("🎛️ Operational Review Threshold Simulator")
    st.markdown(
        "Adjust the decision threshold $\\tau$ to observe how review workload, precision, "
        "and captured churn accounts fluctuate."
    )

    selected_thresh = st.slider("Select Decision Threshold (tau)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    if not tradeoffs.empty:
        # Find closest row
        closest_row = tradeoffs.iloc[(tradeoffs["threshold"] - selected_thresh).abs().argsort()[:1]].iloc[0]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Review Queue Volume", f"{int(closest_row['review_queue_volume']):,} accounts", f"{closest_row['review_share_pct']:.1f}% of test base")
        r2.metric("Captured Churners (TP)", f"{int(closest_row['caught_churn_tp']):,} accounts", f"{closest_row['recall_pct']:.1f}% recall")
        r3.metric("False Alarms (FP)", f"{int(closest_row['false_alarms_fp']):,} accounts", f"{100-closest_row['precision_pct']:.1f}% false discovery")
        r4.metric("Model Precision", f"{closest_row['precision_pct']:.2f}%", f"F1: {closest_row['f1_score']:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Display precomputed evaluation figures
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        roc_img = config.figures_dir / "roc_curve.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Held-Out Test ROC Curve", use_container_width=True)
    with fig_col2:
        pr_img = config.figures_dir / "precision_recall_curve.png"
        if pr_img.exists():
            st.image(str(pr_img), caption="Held-Out Test Precision-Recall Curve", use_container_width=True)

    fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        cm_img = config.figures_dir / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix at Default Threshold (0.50)", use_container_width=True)
    with fig_col4:
        imp_img = config.figures_dir / "feature_importance.png"
        if imp_img.exists():
            st.image(str(imp_img), caption="Permutation Feature Importance (PR-AUC Impact)", use_container_width=True)


def page_action_plan(queue_df: pd.DataFrame) -> None:
    st.header("Retention Action Plan & Prioritized Review Queue")
    st.markdown(
        "Translate quantitative findings into structured, evidence-backed retention interventions. "
        "All recommendations are framed as **testable hypotheses with measurable experiment guardrails**."
    )

    # Action Cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="action-card">
                <h4>1. Early-Tenure Concierge Onboarding</h4>
                <p><b>Observed Evidence:</b> 0–6 month tenure accounts exhibit 52.9% churn prevalence ($49.9K monthly exposure).</p>
                <p><b>Intervention Hypothesis:</b> Proactive 14-day and 30-day onboarding check-ins resolve early configuration blockers.</p>
                <p><b>Business Owner:</b> Customer Success & Support</p>
                <p><b>Primary Metric:</b> 90-day active retention rate (Target: +15% relative lift).</p>
                <p><b>Guardrail:</b> Onboarding support cost per retained subscriber.</p>
                <p><b>Experiment Design:</b> 50/50 randomized trial on new subscribers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-card">
                <h4>2. Annual Plan Renewal Conversion</h4>
                <p><b>Observed Evidence:</b> Month-to-month contracts churn at 42.7% vs 11.3% for One-Year contracts ($120.8K exposure).</p>
                <p><b>Intervention Hypothesis:</b> Modest annual discount incentive stabilizes commitment horizon.</p>
                <p><b>Business Owner:</b> Growth Marketing</p>
                <p><b>Primary Metric:</b> Contract conversion rate and 12-month net retained revenue.</p>
                <p><b>Guardrail:</b> Promotional discount margin dilution.</p>
                <p><b>Experiment Design:</b> In-app discount prompt targeting month 3 M2M accounts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="action-card">
                <h4>3. Fiber Optic Tech Support Trial</h4>
                <p><b>Observed Evidence:</b> Fiber Optic accounts without Tech Support churn at 41.6% ($110.7K exposure).</p>
                <p><b>Intervention Hypothesis:</b> Complimentary 60-day Tech Support trial improves service stability perceptions.</p>
                <p><b>Business Owner:</b> Support Operations</p>
                <p><b>Primary Metric:</b> 6-month retention rate among fiber subscribers.</p>
                <p><b>Guardrail:</b> Support ticket backlog and handle time.</p>
                <p><b>Experiment Design:</b> Bundled support trial offered at installation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-card">
                <h4>4. Automated Payment Migration Credit</h4>
                <p><b>Observed Evidence:</b> Electronic check users churn at 45.3% vs 16.7% for automatic credit card billing ($84.3K exposure).</p>
                <p><b>Intervention Hypothesis:</b> $10 billing credit incentive to adopt automatic bank transfer or credit card auto-pay reduces involuntary churn.</p>
                <p><b>Business Owner:</b> Billing Operations</p>
                <p><b>Primary Metric:</b> Auto-pay migration rate and 6-month retention.</p>
                <p><b>Guardrail:</b> Interchange and processing fees.</p>
                <p><b>Experiment Design:</b> Portal billing reminder test with treatment credit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Prioritized Queue Table
    st.subheader("🎯 Prioritized High-Risk Customer Queue")
    st.markdown(
        "Accounts sorted by **Expected Charge Exposure at Risk** ($\\text{Risk Probability} \\times \\text{Monthly Charges}$). "
        "Allows retention specialists to maximize revenue defense given finite outreach capacity."
    )

    if not queue_df.empty:
        # Filter slider for probability
        prob_filter = st.slider("Filter Minimum Churn Probability", 0.0, 1.0, 0.50, 0.05)
        filtered_queue = queue_df[queue_df["churn_probability"] >= prob_filter]

        st.caption(f"Showing {len(filtered_queue):,} high-risk customer accounts.")
        st.dataframe(
            filtered_queue.style.format({
                "churn_probability": "{:.2%}",
                "MonthlyCharges": "${:.2f}",
                "expected_charge_exposure": "${:.2f}",
            }).set_properties(**{"color": "#1e293b", "background-color": "#ffffff"}),
            use_container_width=True,
            height=340,
        )

        # CSV Download Button
        csv_data = filtered_queue.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Prioritized Queue CSV",
            data=csv_data,
            file_name="prioritized_churn_risk_queue.csv",
            mime="text/csv",
        )


def main() -> None:
    page = render_sidebar()
    df, kpi_summary, seg_summary, tenure_summary, tradeoffs, model_metrics = load_analytical_data()
    queue_df = load_risk_queue()

    if page == "1. Executive Overview":
        page_executive_overview(df, kpi_summary, seg_summary)
    elif page == "2. Segment Explorer":
        page_segment_explorer(df, seg_summary)
    elif page == "3. Lifecycle & Tenure Patterns":
        page_lifecycle_patterns(df, tenure_summary)
    elif page == "4. Predictive Model Diagnostics":
        page_model_evaluation(tradeoffs, model_metrics)
    elif page == "5. Action Plan & Review Queue":
        page_action_plan(queue_df)


if __name__ == "__main__":
    main()
