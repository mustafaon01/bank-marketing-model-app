import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('bank-additional.sav')

# Load data for displaying example
df = pd.read_csv('bank-additional.csv', sep=';')

# Preprocess the data
def preprocess_input(data):
    # Handle missing values
    data = data.replace('unknown', np.nan)
    data = data.replace('nonexistent', np.nan)
    data.fillna(data.mode().iloc[0], inplace=True)

    # Drop unnecessary columns
    data.drop(["day_of_week", "month", "default", "euribor3m"], axis='columns', inplace=True, errors='ignore')

    # Encode categorical variables using one-hot encoding
    categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']
    present_categorical_cols = [col for col in categorical_cols if col in data.columns]
    data = pd.get_dummies(data, columns=present_categorical_cols, drop_first=True)

    # Scale numerical columns
    numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed', 'duration']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data

st.title('Bank Marketing Prediction')

st.write("""
This application uses a Random Forest model to predict whether a customer will subscribe to a term deposit.
""")

# Input form for user data
st.header('Input Customer Data')
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', df['job'].unique())
marital = st.selectbox('Marital', df['marital'].unique())
education = st.selectbox('Education', df['education'].unique())
housing = st.selectbox('Housing Loan', df['housing'].unique())
loan = st.selectbox('Personal Loan', df['loan'].unique())
contact = st.selectbox('Contact Communication Type', df['contact'].unique())
campaign = st.number_input('Number of contacts performed during this campaign', min_value=1, max_value=50, value=1)
pdays = st.number_input('Number of days that passed by after the client was last contacted', min_value=-1, max_value=999, value=999)
previous = st.number_input('Number of contacts performed before this campaign', min_value=0, max_value=50, value=0)
emp_var_rate = st.number_input('Employment Variation Rate', value=1.0)
cons_price_idx = st.number_input('Consumer Price Index', value=93.0)
cons_conf_idx = st.number_input('Consumer Confidence Index', value=-40.0)
nr_employed = st.number_input('Number of Employees', value=5000.0)
duration = st.number_input('Duration of last contact', value=50)

# Create DataFrame for input data
input_data = pd.DataFrame({
    'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'housing': [housing], 'loan': [loan],
    'contact': [contact], 'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx], 'nr.employed': [nr.employed], 'duration': [duration]
})

# Preprocess input data
input_data = preprocess_input(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Yes' if prediction[0] == 'yes' else 'No'
    st.write(f"The model predicts: {result}")
