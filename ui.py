"""
Credit Scoring — Streamlit Frontend
Run: streamlit run ui.py
"""

import streamlit as st
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

FLASK_URL = "http://127.0.0.1:5000"
FIGURES   = "outputs/figures"
METRICS   = "outputs/reports/metrics.json"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        background-color: #3b82d4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #2563eb; }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .good    { background:#d1fae5; color:#065f46; border:2px solid #6ee7b7; }
    .standard{ background:#fef9c3; color:#713f12; border:2px solid #fde047; }
    .poor    { background:#fee2e2; color:#7f1d1d; border:2px solid #fca5a5; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
st.sidebar.title("Credit Scoring App")
page = st.sidebar.radio("Navigation", [
    "Predict Credit Score",
    "Data Analysis Dashboard",
    "Model Performance",
    "About",
])
st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest | Dataset: 100K records")

# ─────────────────────────────────────────────────────────────
# PAGE 1: PREDICTION
# ─────────────────────────────────────────────────────────────
if page == "Predict Credit Score":
    st.title("💳 Credit Score Predictor")
    st.markdown("Fill in the financial details below to predict creditworthiness.")

    # Fetch dropdown options from Flask
    try:
        opts = requests.get(f"{FLASK_URL}/options", timeout=3).json()
        occupations      = opts.get("Occupation", [])
        credit_mixes     = opts.get("Credit_Mix", ["Bad", "Good", "Standard"])
        pay_behaviours   = opts.get("Payment_Behaviour", [])
        pay_min_opts     = opts.get("Payment_of_Min_Amount", ["Yes", "No"])
        backend_alive    = True
    except Exception:
        occupations    = ["Accountant","Architect","Developer","Doctor","Engineer",
                          "Entrepreneur","Journalist","Lawyer","Manager","Mechanic",
                          "Media_Manager","Musician","Scientist","Teacher","Writer"]
        credit_mixes   = ["Bad","Good","Standard"]
        pay_behaviours = ["High_spent_Large_value_payments","High_spent_Medium_value_payments",
                          "High_spent_Small_value_payments","Low_spent_Large_value_payments",
                          "Low_spent_Medium_value_payments","Low_spent_Small_value_payments"]
        pay_min_opts   = ["Yes","No"]
        backend_alive  = False

    if not backend_alive:
        st.warning("Flask backend not detected. Start it with `python app.py` on port 5000.")

    # ── Input Form ───────────────────────────────────────────
    with st.form("credit_form"):
        st.subheader("Personal & Account Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age               = st.number_input("Age", 18, 100, 30)
            occupation        = st.selectbox("Occupation", occupations)
            num_bank_accounts = st.number_input("Number of Bank Accounts", 0, 20, 3)
        with c2:
            annual_income     = st.number_input("Annual Income ($)", 0.0, 5_000_000.0, 45000.0, step=500.0)
            monthly_salary    = st.number_input("Monthly Inhand Salary ($)", 0.0, 50_000.0, 3500.0, step=100.0)
            num_credit_card   = st.number_input("Number of Credit Cards", 0, 30, 4)
        with c3:
            interest_rate     = st.number_input("Interest Rate (%)", 1, 100, 14)
            num_of_loan       = st.number_input("Number of Loans", 0, 20, 3)
            credit_history    = st.number_input("Credit History Age (months)", 0, 500, 200)

        st.subheader("Debt & Payment Information")
        c4, c5, c6 = st.columns(3)
        with c4:
            outstanding_debt  = st.number_input("Outstanding Debt ($)", 0.0, 10_000.0, 1200.0, step=50.0)
            delay_due_date    = st.number_input("Days Delayed from Due Date", -5, 67, 10)
            num_delayed_pay   = st.number_input("Number of Delayed Payments", 0, 100, 5)
        with c5:
            credit_util       = st.slider("Credit Utilization Ratio (%)", 20.0, 50.0, 32.0, step=0.5)
            changed_credit_lim= st.number_input("Changed Credit Limit ($)", -10.0, 40.0, 9.0, step=0.5)
            num_credit_inq    = st.number_input("Number of Credit Inquiries", 0, 30, 5)
        with c6:
            total_emi         = st.number_input("Total EMI per Month ($)", 0.0, 5000.0, 80.0, step=10.0)
            amount_invested   = st.number_input("Amount Invested Monthly ($)", 0.0, 10_000.0, 150.0, step=10.0)
            monthly_balance   = st.number_input("Monthly Balance ($)", 0.0, 2000.0, 350.0, step=10.0)

        st.subheader("Credit Behaviour")
        c7, c8, c9 = st.columns(3)
        with c7:
            credit_mix        = st.selectbox("Credit Mix", credit_mixes)
        with c8:
            payment_behaviour = st.selectbox("Payment Behaviour", pay_behaviours)
        with c9:
            pays_min_amount   = st.selectbox("Pays Only Minimum Amount?", pay_min_opts)

        submitted = st.form_submit_button("Predict Credit Score")

    # ── Predict ──────────────────────────────────────────────
    if submitted:
        payload = {
            "Age":                    age,
            "Annual_Income":          annual_income,
            "Monthly_Inhand_Salary":  monthly_salary,
            "Num_Bank_Accounts":      num_bank_accounts,
            "Num_Credit_Card":        num_credit_card,
            "Interest_Rate":          interest_rate,
            "Num_of_Loan":            num_of_loan,
            "Delay_from_due_date":    delay_due_date,
            "Num_of_Delayed_Payment": num_delayed_pay,
            "Changed_Credit_Limit":   changed_credit_lim,
            "Num_Credit_Inquiries":   num_credit_inq,
            "Credit_Mix":             credit_mix,
            "Outstanding_Debt":       outstanding_debt,
            "Credit_Utilization_Ratio": credit_util,
            "Credit_History_Age":     credit_history,
            "Payment_of_Min_Amount":  pays_min_amount,
            "Total_EMI_per_month":    total_emi,
            "Amount_invested_monthly": amount_invested,
            "Payment_Behaviour":      payment_behaviour,
            "Monthly_Balance":        monthly_balance,
            "Occupation":             occupation,
        }

        with st.spinner("Analyzing credit profile..."):
            try:
                resp = requests.post(f"{FLASK_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Flask backend. Make sure `python app.py` is running.")
                result = None
            except Exception as e:
                st.error(f"Error: {e}")
                result = None

        if result and "credit_score" in result:
            cs      = result["credit_score"]
            conf    = result["confidence"]
            probas  = result["probabilities"]
            css_cls = {"Good": "good", "Standard": "standard", "Poor": "poor"}.get(cs, "standard")
            icon    = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}.get(cs, "ℹ️")

            st.markdown(f"""
            <div class="result-box {css_cls}">
                {icon} Credit Score Prediction: <strong>{cs}</strong><br>
                <span style='font-size:1rem'>Confidence: {conf}%</span>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Class Probabilities")
            prob_cols = st.columns(3)
            prob_colors = {"Good": "#22c55e", "Standard": "#f59e0b", "Poor": "#ef4444"}
            for i, (cls, prob) in enumerate(probas.items()):
                with prob_cols[i]:
                    st.metric(cls, f"{prob*100:.1f}%")
                    st.progress(prob)

            # Credit advice
            st.subheader("Credit Advice")
            if cs == "Good":
                st.success("Excellent credit profile! You are likely to qualify for premium financial products with low interest rates.")
            elif cs == "Standard":
                st.warning("Moderate credit profile. Consider reducing outstanding debt, paying bills on time, and limiting new credit inquiries.")
            else:
                st.error("Poor credit profile. Focus on: paying overdue bills, reducing debt-to-income ratio, and avoiding new credit applications.")

            # Input summary
            with st.expander("View Input Summary"):
                st.json(payload)

# ─────────────────────────────────────────────────────────────
# PAGE 2: DATA ANALYSIS DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "Data Analysis Dashboard":
    st.title("📊 Data Analysis Dashboard")
    st.markdown("Exploratory Data Analysis on 100,000 credit records.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution",
        "Feature Distributions",
        "Features vs Credit Score",
        "Correlation & Behaviour",
    ])

    def show_figure(path, caption=""):
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.info(f"Figure not found: {path}. Run `python scripts/analysis.py` first.")

    with tab1:
        st.subheader("Credit Score Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Good", "17,828", "17.8%")
        with col2:
            st.metric("Standard", "53,174", "53.2%")
        with col3:
            st.metric("Poor", "28,998", "29.0%")
        show_figure(f"{FIGURES}/01_target_distribution.png", "Target Variable Distribution")

    with tab2:
        sub1, sub2 = st.tabs(["Numerical Features", "Categorical Features"])
        with sub1:
            show_figure(f"{FIGURES}/02_numerical_distributions.png", "Numerical Feature Distributions")
        with sub2:
            show_figure(f"{FIGURES}/03_categorical_distributions.png", "Categorical Feature Distributions")

    with tab3:
        sub1, sub2 = st.tabs(["Histogram Overlay", "Box Plots"])
        with sub1:
            show_figure(f"{FIGURES}/04_features_vs_credit_score.png", "Feature Distributions by Credit Score")
        with sub2:
            show_figure(f"{FIGURES}/05_boxplots_key_features.png", "Box Plots by Credit Score")
        show_figure(f"{FIGURES}/12_income_debt_analysis.png", "Income & Debt Analysis")

    with tab4:
        sub1, sub2 = st.tabs(["Correlation Heatmap", "Payment Behaviour"])
        with sub1:
            show_figure(f"{FIGURES}/06_correlation_heatmap.png", "Correlation Heatmap")
        with sub2:
            show_figure(f"{FIGURES}/11_payment_behaviour_vs_credit_score.png", "Payment Behaviour vs Credit Score")

    st.subheader("Key Insights")
    insights = [
        "📌 **Dataset**: 100,000 records, 28 features, 3 credit score classes",
        "📌 **Class imbalance**: Standard (53%) dominates; Good (18%) is underrepresented",
        "📌 **Top predictors**: Outstanding Debt, Interest Rate, Delay from Due Date, Credit Mix",
        "📌 **Missing data**: Credit Mix (20.2%), Monthly Salary (15%), Type of Loan (11.4%)",
        "📌 **Good** customers have lower debt, lower interest rate and longer credit history",
        "📌 **Poor** customers exhibit more delayed payments and higher credit utilization",
        "📌 **Payment Behaviour** strongly correlates with credit score class",
    ]
    for ins in insights:
        st.markdown(ins)

# ─────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("🤖 Model Performance")

    # Load metrics JSON
    metrics_data = None
    if os.path.exists(METRICS):
        with open(METRICS) as f:
            metrics_data = json.load(f)

    model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]

    if metrics_data:
        st.subheader("Performance Comparison")
        rows = []
        for name in model_names:
            if name in metrics_data:
                m = metrics_data[name]
                rows.append({
                    "Model": name,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1']:.4f}",
                    "ROC-AUC": f"{m['auc']:.4f}",
                })
        if rows:
            df_metrics = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df_metrics, use_container_width=True)

        best = metrics_data.get("best_model", "Random Forest")
        st.success(f"Best Model: **{best}** (selected by F1-Score)")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(f"{FIGURES}/09_model_comparison.png"):
            st.image(f"{FIGURES}/09_model_comparison.png", use_column_width=True)
    with col2:
        if os.path.exists(f"{FIGURES}/10_feature_importance.png"):
            st.image(f"{FIGURES}/10_feature_importance.png", use_column_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if os.path.exists(f"{FIGURES}/07_confusion_matrices.png"):
            st.image(f"{FIGURES}/07_confusion_matrices.png", use_column_width=True,
                     caption="Confusion Matrices")
    with col4:
        if os.path.exists(f"{FIGURES}/08_roc_curves.png"):
            st.image(f"{FIGURES}/08_roc_curves.png", use_column_width=True,
                     caption="ROC Curves")

    st.subheader("Top Important Features (Random Forest)")
    top_features = {
        "Outstanding_Debt": 14.89,
        "Interest_Rate": 13.48,
        "Delay_from_due_date": 8.25,
        "Credit_Mix": 8.24,
        "Pays_Min_Only": 6.96,
        "Num_Credit_Inquiries": 4.75,
        "Credit_History_Age": 4.66,
        "Debt_to_Income": 4.48,
        "Num_Credit_Card": 4.40,
        "Changed_Credit_Limit": 3.81,
    }
    df_feat = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance (%)"])
    st.dataframe(df_feat, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4: ABOUT
# ─────────────────────────────────────────────────────────────
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Credit Score Prediction System

    This end-to-end machine learning project predicts a customer's **credit score category**
    (Good, Standard, or Poor) based on their financial profile.

    ### Architecture
    - **Data**: 100,000 training records with 28 features
    - **ML Models**: Logistic Regression, Decision Tree, Random Forest
    - **Best Model**: Random Forest (F1=0.73, AUC=0.87)
    - **Backend**: Flask REST API (`app.py`) — port 5000
    - **Frontend**: Streamlit (`ui.py`) — port 8501

    ### How to Run
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Run data analysis & train models
    python scripts/analysis.py

    # 3. Start Flask backend (new terminal)
    python app.py

    # 4. Start Streamlit frontend (new terminal)
    streamlit run ui.py
    ```

    ### Features Used for Prediction
    | Feature | Description |
    |---------|-------------|
    | Outstanding Debt | Total current debt amount |
    | Interest Rate | Average loan interest rate |
    | Delay from Due Date | Days past payment due |
    | Credit Mix | Type of credit accounts |
    | Credit History Age | Length of credit history (months) |
    | Num Credit Inquiries | Recent credit inquiries count |
    | Payment Behaviour | Spending and payment patterns |
    | Debt to Income Ratio | Outstanding debt / Annual income |

    ### Key Findings
    - **Random Forest** outperformed all models with **72.2% accuracy** and **0.874 ROC-AUC**
    - **Outstanding Debt** and **Interest Rate** are the top two predictors
    - Customers who **pay only the minimum amount** are significantly more likely to have Poor scores
    - **Good** credit customers have 3-4x longer credit history than Poor credit customers
    """)

    st.markdown("---")
    st.caption("Built with Python | scikit-learn | Flask | Streamlit")
