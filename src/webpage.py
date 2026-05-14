import streamlit as st
import numpy as np
import pickle

# load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('Loan Prediction App')

# inputs
no_of_dep = st.slider('No of Dependents', 0, 5)
grad = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_emp = st.selectbox('Self Employed', ['Yes', 'No'])
income = st.number_input('Annual Income')
loan_amount = st.number_input('Loan Amount')
loan_term = st.slider('Loan Term', 0, 30)
cibil = st.slider('CIBIL Score', 0, 1000)
assets = st.number_input('Total Assets Value')

# ✅ CORRECT encoding (MATCHES TRAINING)
grad_s = 1 if grad == 'Graduate' else 0
emp_s = 1 if self_emp == 'Yes' else 0

# prediction
if st.button("Predict"):

    input_data = np.array([[
        no_of_dep,
        grad_s,
        emp_s,
        income,
        loan_amount,
        loan_term,
        cibil,
        assets
    ]])

    # scale
    input_data = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")