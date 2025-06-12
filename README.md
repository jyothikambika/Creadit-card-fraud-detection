# 💳 Credit Card Fraud Detection System

This project aims to detect fraudulent credit card transactions using machine learning techniques. By leveraging real-world datasets and advanced preprocessing methods, the model identifies anomalies in transaction patterns with high accuracy.

## 📌 Project Overview

Credit card fraud is a significant issue in the financial sector. This project uses classification algorithms like *Logistic Regression* and *Decision Trees* to predict fraudulent transactions. The dataset is highly imbalanced, so we apply *SMOTE (Synthetic Minority Oversampling Technique)* to improve model fairness and performance.

## 🚀 Technologies Used

- Python
- scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- SMOTE (from imbalanced-learn)
- Google Cloud Platform (GCP)
- Jupyter Notebook

## 📊 Dataset

- *Source*: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- *Description*: Contains transactions made by European cardholders in September 2013. It has 284,807 transactions, with only 492 (0.17%) being fraudulent.

## 🧠 Model Training

- Preprocessing: Feature scaling, missing value check, correlation matrix
- Imbalance Handling: SMOTE applied to balance the dataset
- Models Used:
  - Logistic Regression
  - Decision Tree Classifier
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Validation: 10-Fold Cross-Validation

## ✅ Results

- *Best Accuracy*: 92%
- *Precision improved by*: 15%
- *Recall improved by*: 20%
- *Prediction latency reduced by*: 30% (after deployment on GCP)

## ☁ Deployment

The trained model was deployed on *Google Cloud Platform*, enabling real-time fraud detection via REST API endpoints.

## 👩‍💻 Role

I was responsible for the entire ML pipeline:
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Evaluation and validation
- Deployment on GCP
- Documentation and GitHub repository setup

## 📂 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ravelajyothi/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
