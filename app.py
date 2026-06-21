"""
app.py
------
Streamlit Web Application for Health Trio Predictor.
Provides an interactive UI for three health prediction models:
  1. Diabetes Prediction (KNN)
  2. Heart Disease Prediction (Decision Tree)
  3. Student Stress Prediction (Naive Bayes)
"""

import streamlit as st
import numpy as np
import pickle
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Trio Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Load Models ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(name):
    """Load a pickled model from the models/ folder."""
    path = os.path.join("models", f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    diabetes_model  = load_model("diabetes_knn")
    diabetes_scaler = load_model("diabetes_scaler")
    heart_model     = load_model("heart_dt")
    heart_scaler    = load_model("heart_scaler")
    stress_model    = load_model("stress_nb")
    stress_scaler   = load_model("stress_scaler")
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏥 Health Trio Predictor")
st.markdown("**AI-powered predictions for Diabetes, Heart Disease & Student Stress**")
st.markdown("---")

if not models_loaded:
    st.error("⚠️ Models not found! Please run `python train.py` first.")
    st.stop()

# ── Sidebar — Choose Predictor ─────────────────────────────────────────────────
st.sidebar.title("🔍 Choose Predictor")
predictor = st.sidebar.radio(
    "Select a health predictor:",
    ["🩺 Diabetes", "❤️ Heart Disease", "😰 Student Stress"]
)

st.sidebar.markdown("---")
st.sidebar.info("Fill in the details on the right and click **Predict** to get your result.")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DIABETES PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if predictor == "🩺 Diabetes":
    st.header("🩺 Diabetes Risk Prediction")
    st.markdown("Model: **K-Nearest Neighbours (from scratch)** | Dataset: Pima Indians Diabetes")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies   = st.slider("Pregnancies", 0, 17, 3)
        glucose       = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)

    with col2:
        insulin       = st.slider("Insulin (mu U/ml)", 0, 846, 79)
        bmi           = st.slider("BMI", 0.0, 67.1, 25.0)
        dpf           = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
        age           = st.slider("Age", 21, 81, 30)

    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **Result: HIGH RISK of Diabetes**")
            st.markdown("Please consult a doctor for proper medical advice.")
        else:
            st.success("✅ **Result: LOW RISK of Diabetes**")
            st.markdown("Keep maintaining a healthy lifestyle!")

        with st.expander("ℹ️ About this Model"):
            st.write("KNN (K=5) implemented from scratch using NumPy. "
                     "Trained on 614 samples, tested on 154 samples. Accuracy: **75.3%**")

# ══════════════════════════════════════════════════════════════════════════════
# 2. HEART DISEASE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif predictor == "❤️ Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    st.markdown("Model: **Decision Tree (from scratch)** | Dataset: Heart Failure UCI")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age         = st.slider("Age", 20, 80, 45)
        sex         = st.selectbox("Sex", ["Male", "Female"])
        chest_pain  = st.selectbox("Chest Pain Type", ["TA (Typical Angina)", "ATA", "NAP", "ASY"])
        resting_bp  = st.slider("Resting Blood Pressure", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 0, 600, 200)
        fbs         = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])

    with col2:
        rest_ecg    = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr      = st.slider("Max Heart Rate", 60, 202, 150)
        ex_angina   = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak     = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)
        st_slope    = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Encode categorical inputs
    sex_val      = 1 if sex == "Male" else 0
    cp_val       = ["TA (Typical Angina)", "ATA", "NAP", "ASY"].index(chest_pain)
    fbs_val      = 1 if fbs == "Yes" else 0
    ecg_val      = ["Normal", "ST", "LVH"].index(rest_ecg)
    angina_val   = 1 if ex_angina == "Yes" else 0
    slope_val    = ["Up", "Flat", "Down"].index(st_slope)

    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        input_data = np.array([[age, sex_val, cp_val, resting_bp, cholesterol,
                                fbs_val, ecg_val, max_hr, angina_val,
                                oldpeak, slope_val]])
        input_scaled = heart_scaler.transform(input_data)
        prediction = heart_model.predict(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **Result: Heart Disease DETECTED**")
            st.markdown("Please consult a cardiologist immediately.")
        else:
            st.success("✅ **Result: No Heart Disease Detected**")
            st.markdown("Your heart indicators look healthy!")

        with st.expander("ℹ️ About this Model"):
            st.write("Decision Tree (max_depth=5) implemented from scratch using NumPy. "
                     "Trained on 734 samples, tested on 184 samples. Accuracy: **80.4%**")

# ══════════════════════════════════════════════════════════════════════════════
# 3. STUDENT STRESS PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif predictor == "😰 Student Stress":
    st.header("😰 Student Stress Level Prediction")
    st.markdown("Model: **Naive Bayes (from scratch)** | Dataset: Student Stress Factors")
    st.markdown("---")

    st.subheader("📊 Rate each factor (0 = Low, 5 = High)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🧠 Psychological**")
        anxiety       = st.slider("Anxiety Level", 0, 5, 2)
        self_esteem   = st.slider("Self Esteem", 0, 5, 3)
        mental_health = st.slider("Mental Health History", 0, 5, 1)
        depression    = st.slider("Depression", 0, 5, 2)

        st.markdown("**📚 Academic**")
        academic_perf = st.slider("Academic Performance", 0, 5, 3)
        study_load    = st.slider("Study Load", 0, 5, 3)
        future_career = st.slider("Future Career Concerns", 0, 5, 3)
        teacher_rel   = st.slider("Teacher-Student Relationship", 0, 5, 3)

        st.markdown("**🌍 Environmental**")
        safety        = st.slider("Safety", 0, 5, 3)
        basic_needs   = st.slider("Basic Needs", 0, 5, 3)

    with col2:
        st.markdown("**🏥 Physiological**")
        sleep         = st.slider("Sleep Quality", 0, 5, 3)
        headache      = st.slider("Headache Frequency", 0, 5, 2)
        blood_pressure_s = st.slider("Blood Pressure", 0, 5, 2)
        breathing     = st.slider("Breathing Problem", 0, 5, 1)
        noise         = st.slider("Noise Level", 0, 5, 2)

        st.markdown("**👥 Social**")
        social_support = st.slider("Social Support", 0, 5, 3)
        peer_pressure  = st.slider("Peer Pressure", 0, 5, 2)
        extracurricular = st.slider("Extracurricular Activities", 0, 5, 2)
        bullying       = st.slider("Bullying", 0, 5, 1)

        living_cond    = st.slider("Living Conditions", 0, 5, 3)

    if st.button("🔍 Predict Stress Level", use_container_width=True):
        input_data = np.array([[anxiety, self_esteem, mental_health, depression,
                                headache, blood_pressure_s, sleep, breathing,
                                noise, living_cond, safety, basic_needs,
                                academic_perf, study_load, teacher_rel, future_career,
                                social_support, peer_pressure, extracurricular, bullying,
                                0]])  # last col placeholder
        
        # Use only first 20 features (model trained on 20 features)
        input_data = input_data[:, :20]
        input_scaled = stress_scaler.transform(input_data)
        prediction = stress_model.predict(input_scaled)[0]

        st.markdown("---")
        labels = {0: ("🟢", "LOW Stress", "success"), 
                  1: ("🟡", "MEDIUM Stress", "warning"),
                  2: ("🔴", "HIGH Stress", "error")}
        
        icon, label, level = labels.get(prediction, ("⚪", "Unknown", "info"))

        if level == "success":
            st.success(f"{icon} **Result: {label}**\nYou're managing stress well. Keep it up!")
        elif level == "warning":
            st.warning(f"{icon} **Result: {label}**\nConsider relaxation techniques and talking to someone.")
        else:
            st.error(f"{icon} **Result: {label}**\nPlease seek support from a counselor or trusted person.")

        with st.expander("ℹ️ About this Model"):
            st.write("Gaussian Naive Bayes implemented from scratch using NumPy. "
                     "Trained on 880 samples, tested on 220 samples. Accuracy: **88.2%**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Health Trio Predictor — CS3703 Programming for AI Project | UET Lahore</small></center>",
    unsafe_allow_html=True
)
