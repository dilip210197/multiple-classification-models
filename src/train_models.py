"""
Train five classifiers for the UCI Bank Marketing dataset.

Before running:
1. Download the UCI Bank Marketing ZIP file.
2. Extract bank-full.csv into this project folder.
3. Run: python train_models.py

This script fits preprocessing ONLY on the training split, saves each complete
preprocessing+classifier pipeline, and exports a held-out test CSV.
"""
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "csv/bank-full.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load the UCI bank-full.csv file (semicolon-separated).
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "bank-full.csv not found. Download and extract the UCI Bank Marketing "
        "dataset, then place bank-full.csv beside train_models.py."
    )

df = pd.read_csv(DATA_PATH, sep=";")
if "y" not in df.columns:
    raise ValueError("Expected target column 'y' in bank-full.csv.")

X = df.drop(columns=["y"])
y = df["y"].astype(str).str.strip().str.lower()

if set(y.unique()) != {"yes", "no"}:
    raise ValueError(f"Expected target labels yes/no; found {sorted(y.unique())}")

# 2. Identify feature types.
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# 3. Split BEFORE fitting preprocessing to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Dense one-hot output is intentional because GaussianNB requires dense input.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_cols),
    ],
    remainder="drop"
)

# LinearSVC is an efficient linear SVM for this dataset size.
# Calibration supplies probabilities so AUC can be calculated consistently.
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
        cv=3
    ),
}

rows = []
for display_name, classifier in models.items():
    print(f"\nTraining {display_name} ...")
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)

    predicted = pipeline.predict(X_test)
    classes = list(pipeline.classes_)
    positive_index = classes.index("yes")
    probabilities = pipeline.predict_proba(X_test)[:, positive_index]

    row = {
        "ML Model": display_name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score((y_test == "yes").astype(int), probabilities),
        "Precision": precision_score(y_test, predicted, pos_label="yes", zero_division=0),
        "Recall": recall_score(y_test, predicted, pos_label="yes", zero_division=0),
        "F1": f1_score(y_test, predicted, pos_label="yes", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    }
    rows.append(row)

    filename = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }[display_name]
    joblib.dump(pipeline, MODEL_DIR / filename)
    print(pd.Series(row).to_string())

# 4. Save held-out test rows WITH target column for app evaluation.
test_data = X_test.copy()
test_data["y"] = y_test
test_data.to_csv(ROOT / "test_data.csv", index=False)

# 5. Save metric comparison table.
results = pd.DataFrame(rows)
results.to_csv(ROOT / "model_comparison.csv", index=False)

print("\n=== FINAL MODEL COMPARISON ===")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nSaved five pipelines in ./model/")
print("Saved test_data.csv and model_comparison.csv")
