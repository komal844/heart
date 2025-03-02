import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_diabetes(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred,prob

# Streamlit UI components
st.title("Heart Diseases Prediction")

# Input fields for each parameter
sex = st.number_input()
cp = st.number_input
trestbps = st.number_input
chol= st.number_input
fbs = st.number_input
restecg = st.number_input
thalach = st.number_input
exang = st.number_input
oldpeak = st.number_input
slope = st.number_input("Slope", min_value=0, max_value=2, value=0, step=1)
ca = st.number_input("CA", min_value=0, max_value=4, value=0, step=1)
thal = st.number_input("Thal", min_value=0, max_value=3, value=0, step=1)

# Create the input dictionary for prediction
input_data = {age': 3,
 'sex': 148,
 'cp': 72,
 'trestbps': 35,
 'chol': 0,
 'fbs': 33.6,
 'restecg': 0.627,
 'thalach': 50,
 'exang': 1,
 'oldpeak': 3.1,
 'slope': 2,
 'ca': 2,
 'thal': 3}
    


# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_heart diseases(input_data)

        if pred == 1:
            st.error(f"Prediction: heart diseases with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No heart diseses with probability {prob:.2f}")
