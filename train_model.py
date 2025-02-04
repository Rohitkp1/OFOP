import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/food_order_data.csv")  # Ensure this file exists

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encoding categorical columns
df['Marital Status'] = label_encoder.fit_transform(df['Marital Status'])  # Encode 'Marital Status'
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])  # Encode 'Occupation'
df['Educational Qualifications'] = label_encoder.fit_transform(df['Educational Qualifications'])  # Encode 'Educational Qualifications'
df['Feedback'] = label_encoder.fit_transform(df['Feedback'])  # Encode 'Feedback'

# Selecting input (X) and output (Y)
X = df[['Age', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'Feedback']]
y = df['WillOrderAgain']  # Target variable

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("model/food_order_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
