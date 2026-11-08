# Bank Marketing – Classification Model Evaluation

## a. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to predict whether a bank customer will subscribe to a term deposit.

The models are evaluated using six performance metrics: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also includes a Streamlit web application that allows users to upload labelled test data, select a trained model, and view its evaluation results. The application loads previously saved models without retraining them.

## b. Dataset Description

**Dataset Name:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information about direct marketing campaigns conducted by a Portuguese banking institution.

- **Number of instances:** 45,211
- **Number of input features:** 16
- **Target variable:** `y`
- **Classification type:** Binary classification
- **Target classes:** `yes` (customer subscribed to a term deposit) and `no` (customer did not subscribe)

The dataset includes numerical and categorical attributes, such as customer age, job, marital status, education, account balance, housing loan status, contact information, and campaign-related variables.

**Data preprocessing:** Numerical features are standardized, and categorical features are encoded using one-hot encoding. The dataset is split into 80% training data and 20% testing data using stratified sampling. Preprocessing is fitted on the training data and saved as part of each model pipeline.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/dilip210197/multiple-classification-models/tree/main

**Deployed Streamlit Application:**  
https://multiple-classification-models-fifoczixvmenyenyhg4uh2.streamlit.app/

The repository contains the training script, Streamlit application, dependency requirements, saved classification models, test dataset, and model comparison results.

## d. Models Used and Performance Comparison

Five classification algorithms were trained and evaluated on the same Bank Marketing dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree | 0.8796 | 0.6961 | 0.4845 | 0.4565 | 0.4701 | 0.4024 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.9010 | 0.9076 | 0.6463 | 0.3403 | 0.4458 | 0.4219 |

All results were obtained from the test dataset after training the models in the BITS Virtual Lab.

## e. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest AUC (0.9079), Recall (0.8147), F1 Score (0.5527), and MCC (0.5092). It identified the largest proportion of actual term-deposit subscribers and provided the strongest overall balance of performance for this task, despite having lower Accuracy and Precision than some other models. |
| **Decision Tree** | Achieved an Accuracy of 0.8796 and an F1 Score of 0.4701. Its AUC of 0.6961 was the lowest among the five models, indicating weaker overall discrimination between the two classes. |
| **KNN** | Achieved relatively high Accuracy (0.8962) and Precision (0.5990), but its Recall was only 0.3403. This indicates that it missed many customers who actually subscribed to a term deposit. |
| **Naive Bayes** | Achieved an Accuracy of 0.8548 and Recall of 0.5198. It provided moderate performance but had the lowest MCC (0.3774), indicating a weaker overall prediction relationship than the other models. |
| **Support Vector Machine** | Achieved the highest Accuracy (0.9010) and Precision (0.6463), along with a strong AUC (0.9076). However, its Recall of 0.3403 was relatively low, meaning that many actual subscribers were not identified. |

## f. Overall Winner for the Dataset

**Winner: Logistic Regression**

Logistic Regression is selected as the overall best-performing model for the Bank Marketing dataset.

Although Support Vector Machine achieved the highest Accuracy of 90.10%, Logistic Regression achieved the highest Recall of 81.47%, F1 Score of 55.27%, MCC of 0.5092, and AUC of 0.9079.

Because the objective is to identify customers likely to subscribe to a term deposit, Recall is particularly important. A model with low Recall may miss many potential subscribers.

Logistic Regression provides the strongest overall performance across the evaluation metrics most relevant to this classification task.

**Conclusion:** Logistic Regression is recommended as the best model for this dataset based on the evaluation results. However, Support Vector Machine may be preferred when maximizing overall Accuracy or Precision is the primary objective.

## g. Streamlit Application

The Streamlit application provides an interactive interface for evaluating saved classification models.

Users can select one of the five classifiers, upload a labelled CSV test dataset, and view the following results:

- Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Confusion matrix
- Classification report

All models and preprocessing steps are loaded from saved pipelines. The application does not retrain models on uploaded test data.

**Live Application:**  
https://multiple-classification-models-fifoczixvmenyenyhg4uh2.streamlit.app/

## h. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- GitHub
- BITS Virtual Lab

## i. Final Conclusion

This project successfully implemented and evaluated five machine learning classification algorithms on the Bank Marketing dataset. The results demonstrate that a model with the highest Accuracy is not necessarily the most suitable model for identifying positive cases.

Logistic Regression performed best overall based on AUC, Recall, F1 Score, and MCC. The deployed Streamlit application enables interactive evaluation of all five trained models using uploaded test data.