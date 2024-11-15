import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
df=pd.read_csv("https://raw.githubusercontent.com/Aparna42mca/Mini-Project/refs/heads/main/bitcoin_price_Training.csv")
df.head()
# Check for any non-numeric characters in 'Volume' and 'Market Cap'
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Remove commas and handle any missing or non-numeric values
df['Volume'] = df['Volume'].replace({'-': '0', ',': ''}, regex=True).astype(float)
df['Market Cap'] = df['Market Cap'].replace({'-': '0', ',': ''}, regex=True).astype(float)

# Display the first few rows after cleaning
df.head()
#  find count of missing values

# Count missing values in each column
missing_values_count = df.isnull().sum()
missing_values_count
#  replace missing values with mean.and print the missing value count

# Replace missing values with the mean for each column
df.fillna(df.mean(), inplace=True)

# Count missing values in each column after filling
missing_values_count_after_fill = df.isnull().sum()
missing_values_count_after_fill
# Features for the model
features = ['Open', 'High', 'Low','Market Cap']
# Target variable
target = 'Close'
X = df[features]
y = df[target]
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

#Save the Model
joblib.dump(model, 'bitcoin_rf_model.pkl')




