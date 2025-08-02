import streamlit as st
import numpy as np
from PIL import Image
from skimage.transform import resize
import joblib

# Title
st.title('Image Classification - Car vs Motorcycle')
st.text('Upload an image and click Predict')

# Load model
model = joblib.load('img_model_compressed.pkl')

# Define categories
CATEGORIES = ['car', 'motorcycle']

# File uploader
uploaded_file = st.file_uploader("Choose a .jpg image", type="jpg")

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')

    # Predict button
    if st.button('PREDICT'):
        st.write('Running prediction...')
        
        # Preprocess
        img = np.array(img)
        img_resized = resize(img, (150, 150, 3))  # keep original training shape
        flat_data = [img_resized.flatten()]
        flat_data = np.array(flat_data)

        # Predict
        y_out = model.predict(flat_data)[0]
        y_proba = model.predict_proba(flat_data)[0]

        # Show result
        st.subheader(f'PREDICTED OUTPUT: {CATEGORIES[y_out]}')
        st.write('Probability:')
        for index, item in enumerate(CATEGORIES):
            st.write(f'{item}: {y_proba[index]*100:.2f}%')
