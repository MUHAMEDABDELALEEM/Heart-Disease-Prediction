main-Epsilon-AI-repo

# 💖 Heart Disease Prediction App

An end-to-end machine learning web app to predict the risk of heart disease based on patient data. Built with Python, trained using real-world clinical data, and deployed using Streamlit.

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection is key to reducing mortality.  
This project aims to build a predictive model that classifies whether a person is at risk of heart disease using various health indicators.

---

## 📊 Dataset Overview

- **Source**: One of the recommended datasets from the Epsilon AI track.
- **Records**: 70,000 rows
- **Features**: 13 original columns (with additional engineered features)
- **Target Variable**: `cardio` (0 = No heart disease, 1 = Risk)

---

## 🧹 Data Cleaning & Preprocessing

- Removed outliers and extreme values (e.g., blood pressure, height, weight)
- Converted `age` from days to years
- Treated inconsistencies and missing values
- Created additional engineered features:
  - `bmi` – Body Mass Index
  - `pulse_pressure` – Difference between systolic and diastolic pressure
  - `risk_score` – Aggregated score based on multiple risk factors
  - `is_obese` – Based on BMI threshold
  - `age_group` – Categorical age segmentation

---

## 📈 Exploratory Data Analysis (EDA)

- Univariate & bivariate analysis
- Correlation heatmaps
- Feature importance ranking using:
  - Random Forest
  - XGBoost
- Statistical summaries and insights

---

## 🧠 Model Building & Evaluation

Multiple classification algorithms were trained and compared:

| Model                | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 72.8%    | 0.70     |
| Random Forest       | 69.3%    | 0.68     |
| **XGBoost (Best)**  | **73.0%**| **0.71** |

### ✅ Final Model

- `XGBoostClassifier`
- Tuned using `GridSearchCV`
- Best Parameters:
  - `learning_rate=0.2`
  - `n_estimators=200`
  - `max_depth=3`

---

## 🧪 Evaluation Metrics

- **Accuracy**: 73%
- **Precision**, **Recall**, **F1 Score**
- Evaluation done on both:
  - Full feature set
  - Selected important features

---

## 🖥️ Streamlit Web Application

### 🔹 Features:

- Real-time prediction interface
- Multilingual support (English / Arabic)
- Clean modern UI with custom CSS and fonts
- Dark Mode enabled
- Combined slider + number inputs
- Dynamic feature engineering in real-time
- Reset button for input values
- Simple and clear result display

### 📷 App Preview:

![App Screenshot](./assets/heart_ui_sample.png) <!-- optional -->

---

## 🗂 Project Structure

📁 heart-disease-prediction/
│
├── heart_data_clean.csv # Cleaned dataset
├── notebook.ipynb # Jupyter Notebook (EDA, modeling)
├── app.py # Streamlit web app code
├── xgboost_heart_model.pkl # Trained model file
├── requirements.txt # Dependencies
└── README.md # Project documentation

---
This project was built as part of the # Epsilon AI Data Science Track Final Project.
Model development, deployment, and documentation by:

MUHAMED ABDEL-ALIM
Machine Learning Engineer

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/heart-disease-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py




