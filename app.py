import streamlit as st
import joblib
import numpy as np

# ========== Streamlit Page Configuration ==========
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💖", layout="wide")

# ========== Custom CSS & Font ==========
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Poppins', sans-serif;
        }
        h1, .title {
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stImage > img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .result {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-top: 30px;
        }
        .footer {
            font-size: 12px;
            text-align: center;
            margin-top: 50px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Load the trained model ==========
model = joblib.load("xgboost_heart_model.pkl")

# ========== Sidebar Language Selection ==========
language = st.sidebar.selectbox("🌐 Language", ["English", "Arabic"])

# ========== Labels for multilingual support ==========
if language == "Arabic":
    labels = {
        "title": "💖 توقع مرض القلب",
        "age": "العمر",
        "gender": "النوع",
        "height": "الطول (سم)",
        "weight": "الوزن (كجم)",
        "ap_hi": "الضغط الانقباضي",
        "ap_lo": "الضغط الانبساطي",
        "chol": "مستوى الكوليسترول",
        "gluc": "مستوى الجلوكوز",
        "smoke": "مدخن",
        "alco": "يتناول الكحول",
        "active": "نشط بدنيًا",
        "predict": "🔍 توقع",
        "reset": "🔄 إعادة تعيين",
        "result_risk": " يوجد خطر الإصابة بمرض القلب!",
        "result_safe": " لا يوجد خطر واضح.",
        "footer": "© 2025 - بواسطة محمد عبد العليم | مهندس تعلم آلي"
    }
    gender_options = ["ذكر", "أنثى"]
    bool_options = ["نعم", "لا"]
    chol_options = ["طبيعي", "فوق الطبيعي", "مرتفع جدًا"]
    gluc_options = ["طبيعي", "فوق الطبيعي", "مرتفع جدًا"]
else:
    labels = {
        "title": "💖 Heart Disease Prediction",
        "age": "Age",
        "gender": "Gender",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "ap_hi": "Systolic BP",
        "ap_lo": "Diastolic BP",
        "chol": "Cholesterol Level",
        "gluc": "Glucose Level",
        "smoke": "Smoker",
        "alco": "Alcohol Intake",
        "active": "Physically Active",
        "predict": "🔍 Predict",
        "reset": "🔄 Reset",
        "result_risk": " Risk of Heart Disease Detected!",
        "result_safe": " No Heart Disease Detected.",
        "footer": "© 2025 - Built by MUHAMED ABDEL-ALIM | ML Engineer"
    }
    gender_options = ["Male", "Female"]
    bool_options = ["Yes", "No"]
    chol_options = ["Normal", "Above Normal", "Well Above Normal"]
    gluc_options = ["Normal", "Above Normal", "Well Above Normal"]

# ========== App Header ==========
st.title(labels["title"])
st.markdown("---")

def dual_input(label, min_val, max_val, default):
    col1, col2 = st.sidebar.columns([2, 1])
    slider_val = col1.slider(label, min_val, max_val, default, key=f"{label}_slider")
    number_val = col2.number_input(label, min_val, max_val, slider_val, key=f"{label}_number")
    return number_val
    
# ========== Sidebar Inputs ==========
st.sidebar.header("🩺 Enter Patient Data")
age = dual_input(labels["age"], 29, 100, 50)
gender = st.sidebar.radio(labels["gender"], gender_options)
height = dual_input(labels["height"], 130, 200, 165)
weight = dual_input(labels["weight"], 40, 150, 70)
ap_hi = dual_input(labels["ap_hi"], 90, 200, 120)
ap_lo = dual_input(labels["ap_lo"], 60, 130, 80)
chol_text = st.sidebar.selectbox(labels["chol"], chol_options)
gluc_text = st.sidebar.selectbox(labels["gluc"], gluc_options)

# ========== Map categorical inputs to numeric ==========
cholesterol = chol_options.index(chol_text) + 1
gluc = gluc_options.index(gluc_text) + 1
smoke = 1 if st.sidebar.radio(labels["smoke"], bool_options) == bool_options[0] else 0
alco = 1 if st.sidebar.radio(labels["alco"], bool_options) == bool_options[0] else 0
active = 1 if st.sidebar.radio(labels["active"], bool_options) == bool_options[0] else 0
gender = 1 if gender == gender_options[0] else 2

# ========== Feature Engineering ==========
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
risk_score = cholesterol + gluc + smoke + alco
is_obese = int(bmi > 30)
age_group = 0 if age < 50 else (1 if age < 60 else 2)

# ========== Feature Array ==========
features = np.array([[gender, height, weight, ap_hi, ap_lo,
                      cholesterol, gluc, smoke, alco, active,
                      bmi, age, age_group, pulse_pressure,
                      risk_score, is_obese]])

# ========== Predict Button ==========
if st.sidebar.button(labels["predict"]):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.markdown(f'<div class="result" style="color:red;">{labels["result_risk"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result" style="color:green;">{labels["result_safe"]}</div>', unsafe_allow_html=True)

# ========== Reset Button ==========
reset = st.sidebar.button(labels["reset"])
if reset:
    st.session_state.clear()
    st.query_params.clear()
    st.success("Form reset successfully. Please re-enter the values.")
    st.stop()
# ========== Footer ==========
st.markdown(f'<div class="footer">{labels["footer"]}</div>', unsafe_allow_html=True)
