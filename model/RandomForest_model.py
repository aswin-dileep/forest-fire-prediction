# model/fuzzy_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
def load_data():
    dataset_path = 'data/Algerian_forest_fires_dataset_CLEANED.csv'
    df = pd.read_csv(dataset_path)
    
    # Basic preprocessing (assuming the dataset has columns like Temperature, Humidity, Wind, etc.)
    # Modify this part based on actual dataset structure
    df = df[['Temperature', 'RH', 'Ws', 'Classes']]  # Adjust this to match your dataset columns
    
    # Feature-target split
    X = df[['Temperature', 'RH', 'Ws']]
    y = df['Classes']
    
    return X, y

# Train the model
def train_model():
    X, y = load_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize a RandomForestClassifier
    model = RandomForestClassifier()
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

# Predict function using trained model
def predict_forest_fire(temperature, humidity, wind_speed):
    # Load the model
    model = train_model()
    
    # Make prediction
    input_data = np.array([[temperature, humidity, wind_speed]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 'fire':
        return "Your forest is in danger!"
    else:
        return "Your forest is safe."
