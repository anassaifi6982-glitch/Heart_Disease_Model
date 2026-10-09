import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Logistic_Regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


st.title("Heart Stroke Prediction by Mohd Anas❤️ ")
st.markdown("Provide the Following details")
age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
Chest_pain = st.selectbox("Chest Pain Type", ["ATA","NAP", "TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)", 80,200,120)
cholosterol = st.number_input("Cholosterol(mg/dL)", 100,600,200)
fasting_bs = st.selectbox("Fasting Blood  Sugar > 120 mg/dL",[0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_HR = st.slider('Max_heart Rate', 60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina",['Y', 'n'])
oldpeak = st.slider("Oldpeak(ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["Up", "Flat", "Down"])

if st.button('Predict'):
    row_input = {
        'Age':age,
        'ReatingBP':resting_bp,
        'Cholesterol':cholosterol,
        'FastingBS':fasting_bs,
        'MaxHR':max_HR,
        'Oldpeak':oldpeak,
        'Sex_'+sex:1,
        'ChestPainType':Chest_pain,
        'RetingECG_'+ resting_ecg:1,
        'ExerciseAngina_ '+ exercise_angina:1,
        'ST_Slope_'+st_slope:1
    }

    input_df = pd.DataFrame([row_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df=input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    # scaled_input = scaler.transfrom(input_df)
    # prediction = model.prediction(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️High Risk of Heart Disese")
    else:
        st.success("✅Low Risk of Heart Disese")

        