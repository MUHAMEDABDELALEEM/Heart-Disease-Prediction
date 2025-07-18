# 💖 Heart Disease Prediction App

This is a machine learning web application that predicts the risk of heart disease based on user input. The model is trained on real-world medical data and deployed using Streamlit.

## 📌 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection can help reduce mortality rates significantly. This project aims to build a predictive model to classify whether a person is at risk of heart disease using health-related features.

## 🔍 Dataset

- Source One of the recommended datasets from the Epsilon AI track.
- Size 70,000 rows × 13 columns.
- Target Variable `cardio` (0 = No heart disease, 1 = Risk detected)
- The dataset required cleaning and preprocessing.

## 🧹 Data Cleaning & Feature Engineering

- Removed and handled outliers in `height`, `weight`, `ap_hi`, and `ap_lo`.
- Converted age to years and created age groups.
- Engineered new features like
  - BMI
  - Pulse Pressure
  - Risk Score
  - Obesity Status

## 📊 Exploratory Data Analysis (EDA)

Performed
- Univariate and bivariate analysis.
- Correlation heatmaps.
- Visualizations using Seaborn and Matplotlib.
- Feature importance analysis using Random Forest.

## ⚙️ Models Tried

- Logistic Regression
- Random Forest
- XGBoost (with hyperparameter tuning using GridSearchCV)

✅ Best Model `XGBoost` with
- Accuracy 73%
- F1 Score 0.71

## 🔬 Evaluation Metrics

Used multiple metrics to evaluate performance
- Accuracy
- Precision
- Recall
- F1 Score

## 🖥️ Web App Deployment

Built an interactive web application using Streamlit
- User can input patient data via sidebar.
- Multilingual support (English & Arabic).
- Responsive and elegant UI with BMI & other features computed live.
- Dark mode support + reset button.

## 🗂 Files Included

- `heart_data_clean.csv` – Cleaned dataset
- `notebook.ipynb` – Main notebook with full analysis
- `app.py` – Streamlit app file
- `xgboost_heart_model.pkl` – Trained model
- `requirements.txt` – Python dependencies
- `README.md` – Project overview

## 🚀 How to Run

To run this app locally

```bash
pip install -r requirements.txt
streamlit run app.py
