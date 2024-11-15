import streamlit as st
import numpy as np
import pandas as pd
import joblib  
from sklearn.ensemble import RandomForestRegressor


# Load the pre-trained Random Forest model
model = joblib.load('bitcoin_rf_model.pkl')
#st.image("bitcoin.jpg", width=200)

# Title and description
st.title("Bitcoin Price Prediction")
st.write("Stay Informed, Stay Ahead Your Guide to Bitcoin's Next Move!")

# Input sliders for the features
open_price = st.number_input("Open Price", min_value=0.0, value=2879.0, step=100.0)
high_price = st.number_input("High Price", min_value=0.0, value=2900.0, step=100.0)
low_price = st.number_input("Low Price", min_value=0.0, value=2850.0, step=100.0)
market_cap = st.number_input("Market Capitalization", min_value=0.0, value=4.55358e10, step=1e9)

# Predict button
if st.button("Predict Closing Price"):
    # Create a DataFrame for the model input
    input_data = pd.DataFrame([[open_price, high_price, low_price, market_cap]], 
                              columns=["Open", "High", "Low", "Market Cap"])
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.write(f"Predicted Closing Price: ${prediction:.2f}")
