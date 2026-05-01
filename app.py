import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Healthcare Risk Predictor")

# Load dataset (auto download)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    return pd.read_csv(url)

df = load_data()

# Train model
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier()
model.fit(X, y)

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

# Prediction
if st.button("Predict"):
    data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
    result = model.predict(data)

    if result[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
