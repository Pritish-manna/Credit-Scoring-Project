"""
Credit Scoring — Flask Backend API
Endpoint: POST /predict
"""

import os, re, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load artefacts ──────────────────────────────────────────
BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

best_model      = joblib.load(os.path.join(MODELS, "best_model.pkl"))
scaler          = joblib.load(os.path.join(MODELS, "scaler.pkl"))
label_encoders  = joblib.load(os.path.join(MODELS, "label_encoders.pkl"))
le_target       = joblib.load(os.path.join(MODELS, "label_encoder_target.pkl"))
medians         = joblib.load(os.path.join(MODELS, "medians.pkl"))
feature_cols    = joblib.load(os.path.join(MODELS, "feature_cols.pkl"))

OCCUPATION_CLASSES       = list(label_encoders["Occupation"].classes_)
CREDIT_MIX_CLASSES       = list(label_encoders["Credit_Mix"].classes_)
PAYMENT_BEHAVIOUR_CLASSES = list(label_encoders["Payment_Behaviour"].classes_)


def _parse_credit_age(val):
    """Parse 'X Years and Y Months' → total months."""
    if val is None or str(val).strip() == "":
        return None
    val = str(val)
    years  = re.search(r"(\d+)\s*Year",  val, re.I)
    months = re.search(r"(\d+)\s*Month", val, re.I)
    y = int(years.group(1))  if years  else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m


def _encode(val, encoder):
    val = str(val).strip()
    if val not in encoder.classes_:
        val = "Unknown"
    return int(encoder.transform([val])[0])


def preprocess(data: dict) -> np.ndarray:
    """
    Transform raw API input dict into scaled feature vector.
    Applies identical pipeline as training (feature engineering,
    encoding, imputation, scaling).
    """
    d = {}

    # ── Raw numeric fields ──────────────────────────────────
    numeric_map = {
        "Annual_Income":           float,
        "Monthly_Inhand_Salary":   float,
        "Num_Bank_Accounts":       float,
        "Num_Credit_Card":         float,
        "Interest_Rate":           float,
        "Num_of_Loan":             float,
        "Delay_from_due_date":     float,
        "Num_of_Delayed_Payment":  float,
        "Changed_Credit_Limit":    float,
        "Num_Credit_Inquiries":    float,
        "Outstanding_Debt":        float,
        "Credit_Utilization_Ratio": float,
        "Total_EMI_per_month":     float,
        "Amount_invested_monthly": float,
        "Monthly_Balance":         float,
        "Age":                     float,
    }
    for col, fn in numeric_map.items():
        raw = data.get(col)
        try:
            d[col] = fn(raw) if raw not in (None, "", "nan") else None
        except (ValueError, TypeError):
            d[col] = None

    # ── Credit History Age ──────────────────────────────────
    raw_cha = data.get("Credit_History_Age")
    if raw_cha is not None:
        # Accept either a number (months) or a string like "22 Years and 3 Months"
        try:
            d["Credit_History_Age"] = float(raw_cha)
        except (ValueError, TypeError):
            d["Credit_History_Age"] = _parse_credit_age(raw_cha)
    else:
        d["Credit_History_Age"] = None

    # ── Categorical fields ──────────────────────────────────
    d["Occupation"]       = _encode(data.get("Occupation",       "Unknown"), label_encoders["Occupation"])
    d["Credit_Mix"]       = _encode(data.get("Credit_Mix",       "Unknown"), label_encoders["Credit_Mix"])
    d["Payment_Behaviour"]= _encode(data.get("Payment_Behaviour","Unknown"), label_encoders["Payment_Behaviour"])

    # ── Payment_of_Min_Amount → Pays_Min_Only ──────────────
    pays_min_raw = str(data.get("Payment_of_Min_Amount", "No")).strip().lower()
    d["Pays_Min_Only"] = 1 if pays_min_raw == "yes" else 0

    # ── Engineered features ─────────────────────────────────
    annual  = d.get("Annual_Income") or 0
    monthly = d.get("Monthly_Inhand_Salary") or 0
    debt    = d.get("Outstanding_Debt") or 0
    emi     = d.get("Total_EMI_per_month") or 0
    invest  = d.get("Amount_invested_monthly") or 0
    delayed = d.get("Num_of_Delayed_Payment") or 0
    util    = d.get("Credit_Utilization_Ratio") or 0

    d["Debt_to_Income"]   = debt / annual  if annual > 0 else None
    d["EMI_to_Income"]    = emi / monthly  if monthly > 0 else None
    d["Investment_Rate"]  = invest / monthly if monthly > 0 else None
    d["Has_Delayed_Payment"] = 1 if delayed > 0 else 0
    d["High_Utilization"]    = 1 if util > 30 else 0

    # ── Build ordered feature vector ────────────────────────
    row = []
    for col in feature_cols:
        val = d.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = float(medians.get(col, 0))
        row.append(float(val))

    arr = np.array(row).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled


# ── Routes ───────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Random Forest Credit Scorer", "classes": list(le_target.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with credit applicant fields.
    Returns prediction + class probabilities.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    try:
        X = preprocess(data)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 422

    try:
        pred_label  = int(best_model.predict(X)[0])
        pred_class  = le_target.inverse_transform([pred_label])[0]
        probas      = best_model.predict_proba(X)[0].tolist()
        classes     = list(le_target.classes_)
        prob_map    = {c: round(p, 4) for c, p in zip(classes, probas)}
        confidence  = round(max(probas) * 100, 2)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "credit_score": pred_class,
        "confidence":   confidence,
        "probabilities": prob_map,
        "model": "Random Forest",
    })


@app.route("/options", methods=["GET"])
def options():
    """Return valid dropdown values for the frontend."""
    return jsonify({
        "Occupation":        OCCUPATION_CLASSES,
        "Credit_Mix":        [c for c in CREDIT_MIX_CLASSES if c != "Unknown"],
        "Payment_Behaviour": [c for c in PAYMENT_BEHAVIOUR_CLASSES if c != "Unknown"],
        "Payment_of_Min_Amount": ["Yes", "No"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
