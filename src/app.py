"""
Streamlit app: evaluate already-trained models on an uploaded labelled test CSV.
The app does NOT fit or retrain any model.
Run locally: streamlit run app.py
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Bank Marketing Model Comparison", page_icon="📊", layout="wide")
st.title("Bank Marketing – Classification Model Evaluation")
st.write(
    "Upload a labelled test CSV containing the original input columns and target "
    "column `y`. Select a saved model to evaluate it. Models are loaded from disk; "
    "no training takes place in this application."
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

selected_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload test_data.csv created by train_models.py to view evaluation results.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if "y" not in test_df.columns:
    st.error("The uploaded CSV must include the target column named 'y'.")
    st.stop()

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"].astype(str).str.strip().str.lower()

if not set(y_test.unique()).issubset({"yes", "no"}):
    st.error("Target column 'y' must contain only 'yes' and/or 'no'.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_name]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train_models.py first.")
    st.stop()

try:
    model = joblib.load(model_path)
    # Pipeline applies its fitted scaler/encoder automatically.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
except Exception as exc:
    st.error(
        "Prediction failed. Check that the uploaded CSV has exactly the same "
        f"feature columns as the training data. Details: {exc}"
    )
    st.stop()

positive_index = list(model.classes_).index("yes")
positive_scores = probabilities[:, positive_index]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label="yes", zero_division=0)
recall = recall_score(y_test, predictions, pos_label="yes", zero_division=0)
f1 = f1_score(y_test, predictions, pos_label="yes", zero_division=0)
mcc = matthews_corrcoef(y_test, predictions)
auc = (
    roc_auc_score((y_test == "yes").astype(int), positive_scores)
    if y_test.nunique() == 2 else float("nan")
)

st.subheader(f"Results – {selected_name}")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}" if pd.notna(auc) else "Not available")
c3.metric("Precision", f"{precision:.4f}")
c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

st.subheader("Confusion matrix")
cm = confusion_matrix(y_test, predictions, labels=["no", "yes"])
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual no", "Actual yes"],
        columns=["Predicted no", "Predicted yes"]
    ),
    use_container_width=True
)

st.subheader("Classification report")
report = classification_report(
    y_test, predictions, labels=["no", "yes"],
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with st.expander("Preview uploaded test data"):
    st.dataframe(test_df.head(20), use_container_width=True)
