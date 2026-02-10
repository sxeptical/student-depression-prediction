import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model & Column List ---
try:
    model = joblib.load('depression_model.pkl')
    model_columns = joblib.load('model_columns.pkl') # <--- NEW: Load the column list
except FileNotFoundError:
    st.error("🚨 Files not found! Make sure you ran Step 1 to save 'model_columns.pkl'.")
    st.stop()

st.title("🧠 Student Mental Health Screening Tool")

# --- 2. User Inputs ---
st.sidebar.header("Student Profile")
academic_pressure = st.sidebar.slider("Academic Pressure (1-5)", 1, 5, 3)
study_hours = st.sidebar.slider("Work/Study Hours (per day)", 0, 16, 6)
financial_stress = st.sidebar.slider("Financial Stress (1-5)", 1, 5, 3)
study_satisfaction = st.sidebar.slider("Study Satisfaction (1-5)", 1, 5, 3)
cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 3.5, step=0.01)
age = st.sidebar.number_input("Age", 18, 35, 21)

# --- 3. Feature Engineering ---
burnout_index = (academic_pressure * study_hours) / (study_satisfaction + 1)
total_stress = financial_stress * academic_pressure

# --- 4. Create Data & Align with Model ---
input_data = pd.DataFrame({
    'Academic Pressure': [academic_pressure],
    'Work/Study Hours': [study_hours],
    'Financial Stress': [financial_stress],
    'Study Satisfaction': [study_satisfaction],
    'CGPA': [cgpa],
    'Age': [age],
    'Burnout_Index': [burnout_index],
    'Total_Stress': [total_stress]
})

# CRITICAL FIX: Add all missing columns (Degree_..., City_...) and set them to 0
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# --- 5. Prediction ---
st.subheader("Assessment Results")
col1, col2 = st.columns(2)
col1.metric("Burnout Index", f"{burnout_index:.2f}")
col2.metric("Total Stress", f"{total_stress:.2f}")

if st.button("Analyze Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ **Low Risk** (Probability: {probability:.2%})")