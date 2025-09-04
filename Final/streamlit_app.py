streamlit_code = '''
import streamlit as st
import joblib
import pandas as pd

st.title('Used Car Price Predictor — Demo')
model = joblib.load('rf_price_pipeline.joblib')

st.sidebar.header('Входные данные')
age = st.sidebar.number_input('Age (years)', 0, 50, 5)
distance_km = st.sidebar.number_input('Distance (km)', 0, 500000, 20000)
brand = st.sidebar.selectbox('Brand', ['other'] + list({}) ) # замените при необходимости
fuel_type = st.sidebar.selectbox('Fuel type', ['petrol','diesel','electric','hybrid','other'])
city = st.sidebar.text_input('City', 'Unknown')

if st.sidebar.button('Predict'):
    X = pd.DataFrame([[age,distance_km,brand,fuel_type,city]], columns=['age','distance_km','brand_top','fuel_type','city'])
    pred = model.predict(X)
    st.write('Predicted price:', pred[0])
'''