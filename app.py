import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

def welcome():
    return "Welcome All"

def predict_note_authentication(tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, MonthlyCharges, TotalCharges,
       MonthlyChargesPerTenure, TotalChargesPerTenure):

    # Load the pre-trained model
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Make predictions on the input data
    prediction = classifier.predict([[tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, MonthlyCharges, TotalCharges,
       MonthlyChargesPerTenure, TotalChargesPerTenure]])

    return prediction

def main():

    html_temp ="""
    <div style="background-color:Blue;padding:10px">
    <h2 style="color:white;text-align:center;">TELECOM CUSTOMER CHURN PREDICTION</h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.image("Automation-in-telecom-industry.jpg")

    tenure = st.number_input('Tenure (Int number (1 -72))', min_value=1, max_value=72, value=1, step=1)
    OnlineSecurity = st.selectbox('OnlineSecurity', [0, 1, 2])
    OnlineBackup = st.selectbox('OnlineBackup', [0, 1, 2, 3])
    DeviceProtection = st.selectbox('DeviceProtection', [0, 1, 2, 3])
    TechSupport = st.selectbox('TechSupport', [0, 1, 2])
    Contract = st.selectbox('Contract', [0, 1, 2])
    MonthlyCharges = st.number_input('MonthlyCharges', value=0)
    TotalCharges = st.number_input('TotalCharges', value=0)
    MonthlyChargesPerTenure = st.number_input('MonthlyChargesPerTenure', value=0)
    TotalChargesPerTenure = st.number_input('TotalChargesPerTenure', value=0)
    result=""
    if st.button("Predict"):
        result = predict_note_authentication(tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, MonthlyCharges, TotalCharges,
       MonthlyChargesPerTenure, TotalChargesPerTenure)

        st.success('The output is {}'.format(result[0]))
        if result[0] == 0:
            st.write("Customer Will Not Churn")
        else:
            st.write("Customer Will Churn")
        
    if st.button("About"):
        st.text("@Eshanya,TuringMinds.ai")
        st.text("Built with Streamlit")

    

if __name__=='__main__':
    main()
