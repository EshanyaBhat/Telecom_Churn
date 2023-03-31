# Importing the required libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Setting the page title and layout
st.set_page_config(page_title="Telecom Churn Prediction")

# Defining the function to load the data
def load_data():
    data = pd.read_csv("train.csv")
    return data

# Defining the function to preprocess the data
def preprocess_data(data):
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, X_test, y_train, y_test, sc_X

# Defining the function to create the Random Forest Model
def create_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model


# Defining the function to get the predictions
def get_predictions(model, sc_X, tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, MonthlyCharges, TotalCharges,
       MonthlyChargesPerTenure, TotalChargesPerTenure):
    data = np.array([[tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
                      TechSupport, Contract, MonthlyCharges, TotalCharges,
                      MonthlyChargesPerTenure, TotalChargesPerTenure]])
    data = sc_X.transform(data)
    prediction = model.predict(data)
    return prediction[0]

# Defining the Streamlit app
def main():

    data = load_data()

    # Preprocessing the data
    X_train, X_test, y_train, y_test, sc_X = preprocess_data(data)

    # Creating the Random Forest Model
    # Creating the Random Forest Model
    model = create_model(X_train, y_train)


    st.title("Telecom Churn Prediction")
    st.image("Automation-in-telecom-industry.jpg")

    # Adding a sidebar to upload the data file
    with st.sidebar:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            X_train, X_test, y_train, y_test, sc_X = preprocess_data(df)
            # Creating the Random Forest Model
            model = create_model(X_train, y_train)

    
    if uploaded_file is not None:
        st.success('File successfully uploaded!')

    st.write('Will the Customer Churn or not?')

    # Create the input form
    tenure = st.number_input('Tenure (Int number (1 -72))', min_value=1, max_value=72, value=1, step=1)
    OnlineSecurity = st.selectbox('OnlineSecurity', [0, 1, 2])
    OnlineBackup = st.selectbox('OnlineBackup', [0, 1, 2])
    DeviceProtection = st.selectbox('DeviceProtection', [0, 1, 2])
    TechSupport = st.selectbox('TechSupport', [0, 1, 2])
    Contract = st.selectbox('Contract', [0, 1, 2])
    MonthlyCharges = st.number_input('MonthlyCharges', value=0)
    TotalCharges = st.number_input('TotalCharges', value=0)
    MonthlyChargesPerTenure = st.number_input('MonthlyChargesPerTenure', value=0)
    TotalChargesPerTenure = st.number_input('TotalChargesPerTenure', value=0)



    # Get the predictions
    if st.button('Predict'):
        result = get_predictions(model, sc_X, tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
                         TechSupport, Contract, MonthlyCharges, TotalCharges,
                         MonthlyChargesPerTenure, TotalChargesPerTenure)

        # Display the results
        st.write('The customer will churn:', result)

    else:
        st.write('The customer will not churn')

if __name__ == '__main__':
    main()
