import streamlit as st
import numpy as np

import joblib as jb
import os

# Title
st.title("Titanic Survival Prediction Using Linear Regression")

# Load the model
survival_model = jb.load("D:\\internship works\\codsoft\\taitanic_survival_prediction.pkl")


# Get the current directory 
BASE_DIR = os.path.dirname(__file__)

# Load the model from the repo
model_path = os.path.join(BASE_DIR, "taitanic_survival_prediction.pkl")
survival_model = jb.load(model_path)


# Input fields
Pclass = st.number_input("Enter the Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)", min_value=1.0, max_value=3.0, value=1.0)
Sex = st.number_input("Enter the Sex (Male = 0, Female = 1)", min_value=0, max_value=1, value=1)
Age = st.number_input("Enter the Age", min_value=1, max_value=100, value=20)
SibSp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=3)
Parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=2)
Fare = st.number_input("Ticket Fare", min_value=0.0, max_value=1000000.0, value=30.0)
Embarked = st.number_input("Embarked Location (Cherbourg = 1, Queenstown = 2, Southampton = 0)", min_value=0, max_value=2, value=1)

# Prediction button
if st.button("Predict Survival"):
    new_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = survival_model.predict(new_data)
    
    st.write(f"🔍 Prediction Input: Pclass={Pclass}, Sex={Sex}, Age={Age}, SibSp={SibSp}, Parch={Parch}, Fare={Fare}, Embarked={Embarked}")
    st.write(f"🧮 Predicted Survival : **{prediction[0]:.2f}**")
    if (prediction ==1):
        st.write("survived")
    else :
     st.write("not survived" )
    st.success("Thank you for using the Titanic Survival Predictor!")




