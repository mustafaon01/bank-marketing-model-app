import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('bank-additional.sav')

df = pd.read_csv('bank-additional.csv', sep=';')

mean_std_values = {
    'age': (40.0, 10.0),
    'campaign': (2.0, 1.0),
    'pdays': (40.0, 100.0),
    'previous': (0.5, 1.0),
    'emp.var.rate': (1.0, 0.5),
    'cons.price.idx': (93.0, 0.5),
    'cons.conf.idx': (-40.0, 5.0),
    'nr.employed': (5000.0, 100.0),
    'duration': (200.0, 100.0)
}

def preprocess_input(data, feature_columns, mean_std_values):
    data = data.replace('unknown', np.nan)
    data = data.replace('nonexistent', np.nan)
    data.fillna(data.mode().iloc[0], inplace=True)

    data.drop(["day_of_week", "month", "default", "euribor3m"], axis='columns', inplace=True, errors='ignore')

    categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']
    present_categorical_cols = [col for col in categorical_cols if col in data.columns]
    data = pd.get_dummies(data, columns=present_categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    for col in mean_std_values:
        mean, std = mean_std_values[col]
        data[col] = (data[col] - mean) / std

    return data

feature_columns = [
    'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed', 'duration',
    'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed',
    'marital_married', 'marital_single', 'education_basic.6y', 'education_basic.9y', 'education_high.school', 'education_illiterate', 'education_professional.course', 'education_university.degree',
    'housing_yes', 'loan_yes', 'contact_telephone'
]

st.title('Bank Marketing Prediction')

st.write("""
This application uses a Random Forest model to predict whether a customer will subscribe to a term deposit.
""")

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

input_data = pd.DataFrame({
    'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'housing': [housing], 'loan': [loan],
    'contact': [contact], 'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx], 'nr.employed': [nr_employed], 'duration': [duration]
})

input_data = preprocess_input(input_data, feature_columns, mean_std_values)

input_data = input_data[feature_columns]

input_data_array = input_data.values

if st.button('Predict'):
    try:
        prediction = model.predict(input_data_array)
        result = 'Yes' if prediction[0] == 'yes' else 'No'
        st.write(f"The model predicts: {result}")
    except ValueError as e:
        st.write(f"Error: {e}")
