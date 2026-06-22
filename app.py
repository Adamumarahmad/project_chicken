import streamlit as st 
from tensorflow.keras.models import load_model

model = load_model('chicken_disease_model.h5') 
st.title('Chicken Disease Detection') 
uploaded_file = st.file_uploader('Upload Chicken Image')
if uploaded_file: 
    st.image(uploaded_file)
if st.button('Predict'): 
    st.write('Prediction Complete')
    