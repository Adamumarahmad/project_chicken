import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #2e3b4e;
    }
    
    /* Optional: Change sidebar text color to make it readable */
    [data-testid="stSidebar"] __element__ { 
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Sidebar")
st.title("Main Content")
class_name=['healthy','sick']
model = load_model('Chicken_disease_model.h5') 
st.title('Chicken Disease Detection') 
uploaded_file = st.file_uploader('Upload Chicken Image')
if uploaded_file: 
    image = Image.open(uploaded_file).convert('RGB')
    st.image(uploaded_file)
if st.button('Predict'): 
    uploaded_img = image.resize((224, 224))
    uploaded_file_to_array = tf.keras.preprocessing.image.img_to_array(uploaded_img)
    uploaded_file_to_array = np.expand_dims(uploaded_file_to_array, axis=0)

    
    prediction = model.predict(uploaded_file_to_array)
    prediction_classes=class_name[np.argmax(prediction)]
    st.write(f'Prediction {prediction_classes}')
    
