
import streamlit as st
import pickle
import numpy as np

# Load the trained model and encoders
with open('model_penguin_66130701910.pkl', 'rb') as f:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(f)

# Title of the app
st.title("Penguin Species Prediction")

# Input features
st.header("Enter the characteristics of the penguin:")

# Input fields for each feature
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, max_value=100.0, step=0.1)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, max_value=30.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0, max_value=300.0, step=1.0)
body_mass = st.number_input("Body Mass (g)", min_value=0.0, max_value=8000.0, step=10.0)

# Dropdown for island selection
island = st.selectbox("Island", ['Biscoe', 'Dream', 'Torgersen'])
island_encoded = island_encoder.transform([island])[0]

# Dropdown for sex selection
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])
sex_encoded = sex_encoder.transform([sex])[0]

# Make a prediction
if st.button("Predict"):
    # Prepare input data for the model
    input_data = np.array([[culmen_length, culmen_depth, flipper_length, body_mass, island_encoded, sex_encoded]])

    # Make prediction
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)

    # Display the result
    st.success(f"The predicted species is: {predicted_species}")
