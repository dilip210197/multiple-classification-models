# Machine Learning Assignment 2 – Bank Marketing Classification

> Learning starter project. Run the experiment yourself, verify the outputs, and write your own interpretation. Do not submit placeholder values or unverified claims.

## 1. Problem statement
The objective is to predict whether a bank customer will subscribe to a term deposit (`yes`/`no`) using customer and campaign attributes.

## 2. Dataset description
- Dataset: UCI Bank Marketing (`bank-full.csv`)
- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- File format: CSV separated by semicolons (`;`)
- Target column: `y`
- The commonly distributed `bank-full.csv` contains 45,211 records and 16 input features plus the target. Confirm the downloaded file's dimensions in your own run.

## 3. GitHub repository link
Replace this line with your actual public repository URL: `PASTE_GITHUB_URL_HERE`

## 4. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine (linear SVM with probability calibration)

## 5. Model evaluation
Run `train_models.py`. Copy the actual values from `model_comparison.csv` into the table below.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

## 6. Observations
Write your own observations using your actual metrics. Discuss accuracy, AUC, precision, recall, F1 and MCC, and explain any trade-offs.

- Logistic Regression: [write your result-based observation]
- Decision Tree: [write your result-based observation]
- KNN: [write your result-based observation]
- Naive Bayes: [write your result-based observation]
- Support Vector Machine: [write your result-based observation]

## 7. Overall winner
[Name the model you select and justify it using your measured metrics. Consider the class imbalance and the trade-off between precision and recall; do not choose by accuracy alone.]

## 8. Streamlit application
Replace this line with your actual deployed app URL: `PASTE_STREAMLIT_APP_URL_HERE`

## 9. Technologies used
Python, pandas, scikit-learn, joblib and Streamlit.

## How to run
1. Download and extract the UCI dataset.
2. Place `bank-full.csv` in the same folder as `train_models.py`.
3. Install dependencies: `pip install -r requirements.txt`
4. Train and export models: `python train_models.py`
5. Start the web app: `streamlit run app.py`
6. Upload the generated `test_data.csv` in the app.

## Project files
- `train_models.py`: fits preprocessing and the five classifiers, evaluates them, and saves complete pipelines.
- `app.py`: loads saved pipelines and evaluates an uploaded labelled test CSV. It does not retrain.
- `model/`: saved model pipelines (generated after training).
- `test_data.csv`: held-out test data with target labels (generated after training).
- `model_comparison.csv`: evaluation table (generated after training).
