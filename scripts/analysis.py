"""
Credit Scoring - Data Analysis & Machine Learning Script
Covers: EDA, Feature Engineering, Model Training, Evaluation, Visualization
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, re, os, json, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

FIGURES = "outputs/figures"
MODELS  = "models"
REPORTS = "outputs/reports"
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS,  exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATASETS")
print("=" * 60)

train_raw = pd.read_csv("data/train.csv", low_memory=False)
test_raw  = pd.read_csv("data/test.csv",  low_memory=False)

print(f"Train shape : {train_raw.shape}")
print(f"Test  shape : {test_raw.shape}")
print(f"\nColumns : {list(train_raw.columns)}")

# ─────────────────────────────────────────────
# 2. INSPECT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DATASET INSPECTION")
print("=" * 60)
print("\nTrain dtypes:\n", train_raw.dtypes)
print("\nTrain head:\n", train_raw.head(3).to_string())
print("\nTest head:\n", test_raw.head(3).to_string())

# ─────────────────────────────────────────────
# 3. CLEAN HELPER FUNCTIONS
# ─────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace from string columns
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Replace common placeholders with NaN
    PLACEHOLDERS = ["nan", "NaN", "NA", "N/A", "null", "NULL", "", "_", "__", "!@9#%8", "#F%$D@*&8"]
    for c in df.columns:
        df[c] = df[c].replace(PLACEHOLDERS, np.nan)

    # ── Annual_Income: remove trailing underscores/letters ──
    if "Annual_Income" in df.columns:
        df["Annual_Income"] = df["Annual_Income"].astype(str).str.replace(r"[_a-zA-Z]+$", "", regex=True)
        df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")

    # ── Monthly_Inhand_Salary ──
    if "Monthly_Inhand_Salary" in df.columns:
        df["Monthly_Inhand_Salary"] = pd.to_numeric(df["Monthly_Inhand_Salary"], errors="coerce")

    # ── Age: remove non-numeric chars, clamp 18-100 ──
    if "Age" in df.columns:
        df["Age"] = df["Age"].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df["Age"].apply(lambda x: x if (pd.notna(x) and 18 <= x <= 100) else np.nan)

    # ── Num_of_Loan ──
    if "Num_of_Loan" in df.columns:
        df["Num_of_Loan"] = df["Num_of_Loan"].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
        df["Num_of_Loan"] = pd.to_numeric(df["Num_of_Loan"], errors="coerce")
        df["Num_of_Loan"] = df["Num_of_Loan"].clip(lower=0)

    # ── Num_of_Delayed_Payment ──
    if "Num_of_Delayed_Payment" in df.columns:
        df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
        df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"], errors="coerce")
        df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].clip(lower=0)

    # ── Changed_Credit_Limit ──
    if "Changed_Credit_Limit" in df.columns:
        df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
        df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"], errors="coerce")

    # ── Num_Credit_Inquiries ──
    if "Num_Credit_Inquiries" in df.columns:
        df["Num_Credit_Inquiries"] = pd.to_numeric(df["Num_Credit_Inquiries"], errors="coerce")
        df["Num_Credit_Inquiries"] = df["Num_Credit_Inquiries"].clip(lower=0)

    # ── Outstanding_Debt ──
    if "Outstanding_Debt" in df.columns:
        df["Outstanding_Debt"] = df["Outstanding_Debt"].astype(str).str.replace(r"[^0-9\.]", "", regex=True)
        df["Outstanding_Debt"] = pd.to_numeric(df["Outstanding_Debt"], errors="coerce")

    # ── Credit_History_Age -> months ──
    if "Credit_History_Age" in df.columns:
        def parse_credit_age(val):
            if pd.isna(val):
                return np.nan
            val = str(val)
            years  = re.search(r"(\d+)\s*Year",  val, re.I)
            months = re.search(r"(\d+)\s*Month", val, re.I)
            y = int(years.group(1))  if years  else 0
            m = int(months.group(1)) if months else 0
            return y * 12 + m
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(parse_credit_age)

    # ── Amount_invested_monthly ──
    if "Amount_invested_monthly" in df.columns:
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype(str).str.replace(r"[^0-9\.]", "", regex=True)
        df["Amount_invested_monthly"] = pd.to_numeric(df["Amount_invested_monthly"], errors="coerce")

    # ── Monthly_Balance ──
    if "Monthly_Balance" in df.columns:
        df["Monthly_Balance"] = df["Monthly_Balance"].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
        df["Monthly_Balance"] = pd.to_numeric(df["Monthly_Balance"], errors="coerce")

    # ── Credit_Mix: consolidate categories ──
    if "Credit_Mix" in df.columns:
        valid_mix = {"Good", "Standard", "Bad"}
        df["Credit_Mix"] = df["Credit_Mix"].apply(lambda x: x if x in valid_mix else np.nan)

    return df


train = clean_dataframe(train_raw)
test  = clean_dataframe(test_raw)

# ─────────────────────────────────────────────
# 4. BASIC STATISTICS & MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: BASIC STATISTICS & MISSING VALUES")
print("=" * 60)

num_cols = train.select_dtypes(include=np.number).columns.tolist()
print("\nNumerical Statistics:\n", train[num_cols].describe().round(2).to_string())

missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Pct%": missing_pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Pct%", ascending=False)
print(f"\nMissing values:\n{missing_df.to_string()}")

dups = train.duplicated().sum()
print(f"\nDuplicate rows: {dups}")

# ─────────────────────────────────────────────
# 5. TARGET DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TARGET VARIABLE — Credit_Score")
print("=" * 60)
target_dist = train["Credit_Score"].value_counts()
print(target_dist)
target_pct  = (target_dist / len(train) * 100).round(2)
print(target_pct)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4caf50", "#ff9800", "#f44336"]
ax.bar(target_dist.index, target_dist.values, color=colors, edgecolor="white", width=0.5)
for i, (val, pct) in enumerate(zip(target_dist.values, target_pct.values)):
    ax.text(i, val + 300, f"{val:,}\n({pct}%)", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Credit Score Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Credit Score Category"); ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(f"{FIGURES}/01_target_distribution.png", dpi=150)
plt.close()
print("-> Saved 01_target_distribution.png")

# ─────────────────────────────────────────────
# 6. EDA — NUMERICAL DISTRIBUTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: EDA — NUMERICAL DISTRIBUTIONS")
print("=" * 60)

eda_num_cols = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Credit_History_Age", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]
eda_num_cols = [c for c in eda_num_cols if c in train.columns]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(eda_num_cols):
    data = train[col].dropna()
    axes[i].hist(data, bins=40, color="#3b82d4", edgecolor="white", alpha=0.8)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Count")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES}/02_numerical_distributions.png", dpi=150)
plt.close()
print("-> Saved 02_numerical_distributions.png")

# ─────────────────────────────────────────────
# 7. EDA — CATEGORICAL FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: EDA — CATEGORICAL FEATURES")
print("=" * 60)

cat_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Month"]
cat_cols = [c for c in cat_cols if c in train.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc = train[col].value_counts().head(12)
    axes[i].barh(vc.index[::-1], vc.values[::-1], color="#7c5cd8", edgecolor="white")
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Count")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/03_categorical_distributions.png", dpi=150)
plt.close()
print("-> Saved 03_categorical_distributions.png")

# ─────────────────────────────────────────────
# 8. EDA — FEATURES vs CREDIT SCORE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: KEY FEATURES vs CREDIT SCORE")
print("=" * 60)

key_num = ["Annual_Income", "Outstanding_Debt", "Credit_History_Age",
           "Num_of_Delayed_Payment", "Interest_Rate", "Credit_Utilization_Ratio",
           "Delay_from_due_date", "Monthly_Balance"]
key_num = [c for c in key_num if c in train.columns]

order = ["Poor", "Standard", "Good"]
order = [o for o in order if o in train["Credit_Score"].unique()]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
palette = {"Good": "#4caf50", "Standard": "#ff9800", "Poor": "#f44336"}

for i, col in enumerate(key_num):
    data = train[[col, "Credit_Score"]].dropna()
    for cs in order:
        subset = data[data["Credit_Score"] == cs][col]
        axes[i].hist(subset, bins=30, alpha=0.6, label=cs, color=palette.get(cs, "gray"))
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].legend(fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distribution by Credit Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/04_features_vs_credit_score.png", dpi=150)
plt.close()
print("-> Saved 04_features_vs_credit_score.png")

# Box plots for key features
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for i, col in enumerate(key_num):
    data = train[[col, "Credit_Score"]].dropna()
    groups = [data[data["Credit_Score"] == cs][col].values for cs in order]
    bp = axes[i].boxplot(groups, tick_labels=order, patch_artist=True, notch=False)
    for patch, cs in zip(bp["boxes"], order):
        patch.set_facecolor(palette.get(cs, "lightgray"))
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].tick_params(axis="x", labelsize=9)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Box Plots: Key Features by Credit Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/05_boxplots_key_features.png", dpi=150)
plt.close()
print("-> Saved 05_boxplots_key_features.png")

# ─────────────────────────────────────────────
# 9. CORRELATION HEATMAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: CORRELATION HEATMAP")
print("=" * 60)

corr_cols = [c for c in eda_num_cols if c in train.columns]
corr_matrix = train[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, annot_kws={"size": 8},
            vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap — Numerical Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/06_correlation_heatmap.png", dpi=150)
plt.close()
print("-> Saved 06_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 10. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: FEATURE ENGINEERING")
print("=" * 60)

DROP_COLS = ["ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Debt-to-Income ratio
    df["Debt_to_Income"] = np.where(
        df["Annual_Income"] > 0,
        df["Outstanding_Debt"] / df["Annual_Income"],
        np.nan
    )

    # EMI burden ratio
    df["EMI_to_Income"] = np.where(
        df["Monthly_Inhand_Salary"] > 0,
        df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"],
        np.nan
    )

    # Investment rate
    df["Investment_Rate"] = np.where(
        df["Monthly_Inhand_Salary"] > 0,
        df["Amount_invested_monthly"] / df["Monthly_Inhand_Salary"],
        np.nan
    )

    # Has delayed payment flag
    df["Has_Delayed_Payment"] = (df["Num_of_Delayed_Payment"].fillna(0) > 0).astype(int)

    # High utilization flag
    df["High_Utilization"] = (df["Credit_Utilization_Ratio"] > 30).astype(int)

    # Payment min amount binary
    if "Payment_of_Min_Amount" in df.columns:
        df["Pays_Min_Only"] = (df["Payment_of_Min_Amount"].astype(str).str.strip().str.lower() == "yes").astype(int)
        df.drop(columns=["Payment_of_Min_Amount"], inplace=True)

    # Drop raw identifiers
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    return df


train_fe = engineer_features(train)
test_fe  = engineer_features(test)
print(f"Features after engineering — train: {train_fe.shape[1]}, test: {test_fe.shape[1]}")

# ─────────────────────────────────────────────
# 11. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: ENCODING CATEGORICAL FEATURES")
print("=" * 60)

cat_encode = ["Occupation", "Credit_Mix", "Payment_Behaviour"]
cat_encode = [c for c in cat_encode if c in train_fe.columns]

label_encoders = {}
for col in cat_encode:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0).fillna("Unknown").astype(str)
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].fillna("Unknown").astype(str))
    test_fe[col]  = le.transform(test_fe[col].fillna("Unknown").astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# Encode target
le_target = LabelEncoder()
train_fe["Credit_Score"] = le_target.fit_transform(train_fe["Credit_Score"].fillna("Standard"))
print(f"Target classes: {list(le_target.classes_)}")

# ─────────────────────────────────────────────
# 12. PREPARE FEATURES / IMPUTE / SCALE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: PREPARE FEATURES, IMPUTE, SCALE")
print("=" * 60)

TARGET = "Credit_Score"
feature_cols = [c for c in train_fe.columns if c != TARGET and c in test_fe.columns]

X = train_fe[feature_cols].copy()
y = train_fe[TARGET].copy()
X_test_final = test_fe[feature_cols].copy()

# Impute with median (from training only — prevent leakage)
medians = X.median()
X.fillna(medians, inplace=True)
X_test_final.fillna(medians, inplace=True)

# Scale
scaler = StandardScaler()
X_scaled       = scaler.fit_transform(X)
X_test_scaled  = scaler.transform(X_test_final)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Target distribution: {dict(zip(le_target.classes_, np.bincount(y)))}")

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ─────────────────────────────────────────────
# 13. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: MODEL TRAINING")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, solver="lbfgs",
        C=1.0, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=30, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        n_jobs=-1, random_state=42, class_weight="balanced"
    ),
}

results = {}
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=le_target.classes_))

# ─────────────────────────────────────────────
# 14. CONFUSION MATRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_val, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_, ax=ax,
                linewidths=0.5)
    ax.set_title(f"{name}\n(Acc={res['accuracy']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/07_confusion_matrices.png", dpi=150)
plt.close()
print("-> Saved 07_confusion_matrices.png")

# ─────────────────────────────────────────────
# 15. ROC CURVES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 14: ROC CURVES")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
line_colors = ["#3b82d4", "#f44336", "#4caf50"]
for ax, (name, res) in zip(axes, results.items()):
    for ci, (cls_name, color) in enumerate(zip(le_target.classes_, line_colors)):
        y_bin = (y_val == ci).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, res["y_proba"][:, ci])
        auc_ci = roc_auc_score(y_bin, res["y_proba"][:, ci])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={auc_ci:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC — {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
plt.suptitle("ROC Curves — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/08_roc_curves.png", dpi=150)
plt.close()
print("-> Saved 08_roc_curves.png")

# ─────────────────────────────────────────────
# 16. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 15: MODEL COMPARISON")
print("=" * 60)

metrics = ["accuracy", "precision", "recall", "f1", "auc"]
comp_df = pd.DataFrame(
    {name: [res[m] for m in metrics] for name, res in results.items()},
    index=metrics
).T.round(4)
print(comp_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics))
width = 0.25
bar_colors = ["#3b82d4", "#ff9800", "#4caf50"]
for i, (name, row) in enumerate(comp_df.iterrows()):
    ax.bar(x + i * width, row.values, width, label=name, color=bar_colors[i], edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend()
for i, (name, row) in enumerate(comp_df.iterrows()):
    for j, v in enumerate(row.values):
        ax.text(j + i * width, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURES}/09_model_comparison.png", dpi=150)
plt.close()
print("-> Saved 09_model_comparison.png")

# ─────────────────────────────────────────────
# 17. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 16: FEATURE IMPORTANCE — Random Forest")
print("=" * 60)

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
top_features = feat_imp.head(15)
print(top_features.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))[::-1]
ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors_imp, edgecolor="white")
ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIGURES}/10_feature_importance.png", dpi=150)
plt.close()
print("-> Saved 10_feature_importance.png")

# ─────────────────────────────────────────────
# 18. SELECT & SAVE BEST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 17: SELECT & SAVE BEST MODEL")
print("=" * 60)

best_name = comp_df["f1"].idxmax()
best_model = results[best_name]["model"]
print(f"Best model: {best_name} (F1={comp_df.loc[best_name, 'f1']:.4f})")

# Re-train on FULL training data
best_model.fit(X_scaled, y)

joblib.dump(best_model,    f"{MODELS}/best_model.pkl")
joblib.dump(scaler,        f"{MODELS}/scaler.pkl")
joblib.dump(label_encoders, f"{MODELS}/label_encoders.pkl")
joblib.dump(le_target,     f"{MODELS}/label_encoder_target.pkl")
joblib.dump(medians,       f"{MODELS}/medians.pkl")
joblib.dump(feature_cols,  f"{MODELS}/feature_cols.pkl")

print("Saved: best_model.pkl, scaler.pkl, label_encoders.pkl, label_encoder_target.pkl, medians.pkl, feature_cols.pkl")

# ─────────────────────────────────────────────
# 19. SAVE METRICS JSON
# ─────────────────────────────────────────────
metrics_out = {}
for name, res in results.items():
    metrics_out[name] = {k: round(float(v), 4) for k, v in res.items() if isinstance(v, float)}
metrics_out["best_model"] = best_name
metrics_out["classes"] = list(le_target.classes_)
metrics_out["feature_cols"] = feature_cols

with open(f"{REPORTS}/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nSaved metrics to {REPORTS}/metrics.json")

# ─────────────────────────────────────────────
# 20. PAYMENT BEHAVIOUR ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 18: PAYMENT BEHAVIOUR vs CREDIT SCORE")
print("=" * 60)

if "Payment_Behaviour" in train.columns:
    pb_cs = train.groupby(["Payment_Behaviour", "Credit_Score"]).size().unstack(fill_value=0)
    pb_cs_pct = pb_cs.div(pb_cs.sum(axis=1), axis=0) * 100
    pb_cs_pct = pb_cs_pct.head(8)

    fig, ax = plt.subplots(figsize=(12, 5))
    pb_cs_pct.plot(kind="bar", ax=ax, color=["#f44336", "#ff9800", "#4caf50"], edgecolor="white")
    ax.set_title("Payment Behaviour vs Credit Score (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Payment Behaviour"); ax.set_ylabel("Percentage (%)")
    ax.legend(title="Credit Score"); ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/11_payment_behaviour_vs_credit_score.png", dpi=150)
    plt.close()
    print("-> Saved 11_payment_behaviour_vs_credit_score.png")

# ─────────────────────────────────────────────
# 21. INCOME & DEBT ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 19: INCOME & DEBT ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cs, color in palette.items():
    subset = train[train["Credit_Score"] == cs]["Annual_Income"].dropna()
    axes[0].hist(subset, bins=30, alpha=0.6, label=cs, color=color)
axes[0].set_title("Annual Income by Credit Score", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Annual Income"); axes[0].legend()

for cs, color in palette.items():
    subset = train[train["Credit_Score"] == cs]["Outstanding_Debt"].dropna()
    axes[1].hist(subset, bins=30, alpha=0.6, label=cs, color=color)
axes[1].set_title("Outstanding Debt by Credit Score", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Outstanding Debt"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/12_income_debt_analysis.png", dpi=150)
plt.close()
print("-> Saved 12_income_debt_analysis.png")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE")
print(f"Figures saved to: {FIGURES}")
print(f"Models saved to : {MODELS}")
print(f"Reports saved to: {REPORTS}")
print("=" * 60)
