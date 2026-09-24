# Credit Scoring Project

An end-to-end Machine Learning project that predicts a customer's **credit score** (Good / Standard / Poor) from their financial profile, complete with a Flask REST API and a Streamlit web UI.

---

## Project Structure

```
Credit_scoring_project/
├── data/
│   ├── train.csv               # 100,000 training records
│   └── test.csv                # 50,000 test records
├── models/
│   ├── best_model.pkl          # Trained Random Forest
│   ├── scaler.pkl              # StandardScaler
│   ├── label_encoders.pkl      # Categorical encoders
│   ├── label_encoder_target.pkl
│   ├── medians.pkl             # Imputation medians
│   └── feature_cols.pkl        # Ordered feature list
├── outputs/
│   ├── figures/                # 12 EDA & model charts
│   └── reports/
│       └── metrics.json        # Model performance metrics
|       └── PritishManna_CreditScoring_ProjectReport.docx        
├── scripts/
│   └── analysis.py             # Full EDA + ML pipeline script
├── notebooks/
│   └── PritishManna_CreditScoring.ipynb    # Jupyter Notebook
├── app.py                      # Flask backend API
├── ui.py                       # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run data analysis and train models
```bash
python scripts/analysis.py
```
This will:
- Clean and preprocess 100K records
- Generate 12 EDA visualizations in `outputs/figures/`
- Train Logistic Regression, Decision Tree, Random Forest
- Save the best model (Random Forest) and all artefacts to `models/`
- Save `outputs/reports/metrics.json`

---

## Running the Application

### Step 1 — Start Flask backend (Terminal 1)
```bash
python app.py
```
API will be available at `http://127.0.0.1:5000`

### Step 2 — Start Streamlit frontend (Terminal 2)
```bash
streamlit run ui.py
```
UI will open at `http://localhost:8501`

---

## API Reference

### `GET /`
Health check — returns model info.

### `GET /options`
Returns valid dropdown values for all categorical fields.

### `POST /predict`
Predict credit score from applicant data.

**Request body (JSON):**
```json
{
  "Age": 30,
  "Annual_Income": 55000,
  "Monthly_Inhand_Salary": 4200,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 4,
  "Interest_Rate": 14,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 3,
  "Changed_Credit_Limit": 9.5,
  "Num_Credit_Inquiries": 4,
  "Credit_Mix": "Good",
  "Outstanding_Debt": 1100,
  "Credit_Utilization_Ratio": 28,
  "Credit_History_Age": 200,
  "Payment_of_Min_Amount": "No",
  "Total_EMI_per_month": 80,
  "Amount_invested_monthly": 150,
  "Payment_Behaviour": "Low_spent_Small_value_payments",
  "Monthly_Balance": 350,
  "Occupation": "Engineer"
}
```

**Response:**
```json
{
  "credit_score": "Good",
  "confidence": 74.32,
  "probabilities": {
    "Good": 0.7432,
    "Poor": 0.0821,
    "Standard": 0.1747
  },
  "model": "Random Forest"
}
```

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6089   | 0.6068    | 0.6089 | 0.5939   | 0.7429  |
| Decision Tree       | 0.7123   | 0.7163    | 0.7123 | 0.7137   | 0.8359  |
| **Random Forest**   | **0.7216** | **0.7635** | **0.7216** | **0.7257** | **0.8740** |

**Best model**: Random Forest ✅

---

## Key Insights

- **Top predictors**: Outstanding Debt (14.9%), Interest Rate (13.5%), Delay from Due Date (8.3%), Credit Mix (8.2%)
- Customers paying **only the minimum amount** are 3x more likely to have a Poor score
- **Good** credit customers have 3-4x longer credit history on average
- Dataset has class imbalance: Standard (53%), Poor (29%), Good (18%)

---

## Technologies

- **Python 3.12**
- **scikit-learn** — ML models
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Visualizations
- **Flask** — REST API backend
- **Streamlit** — Interactive web UI
- **joblib** — Model serialization
