import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('Chicken_disease_model.h5') 
st.title('Chicken Disease Detection') 
uploaded_file = st.file_uploader('Upload Chicken Image')
if uploaded_file: 
    st.image(uploaded_file)
if st.button('Predict'): 
    uploaded_file = uploaded_file.resize(224, 224)
    uploaded_file_to_array = tf.keras.preprocessing.image.uploaded_file_to_array(uploaded_file)
    uploaded_file_to_array = np.expand(uploaded_file_to_array, axis=0)
        
    prediction = model.predict(uploaded_file_to_array)
    st.write('Prediction Complete')
    
