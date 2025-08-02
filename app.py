import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image

st.title('Image Classification')
st.text('Upload an image to classify it as a car or motorcycle')


model = pickle.load(open('img_model.p', 'rb'))
CATEGORIES = ['car', 'motorcycle']


uploaded_file = st.file_uploader("Choose a JPG image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')

    # Predict on button click
    if st.button('PREDICT'):
        try:
            img = np.array(img)
            img_resized = resize(img, (150, 150, 3))  # resize to model input shape
            flat_data = [img_resized.flatten()]
            flat_data = np.array(flat_data)

            # Prediction
            y_out = model.predict(flat_data)
            prediction_label = CATEGORIES[y_out[0]]
            st.success(f'PREDICTED OUTPUT: {prediction_label}')

            # Probability
            probabilities = model.predict_proba(flat_data)[0]
            for index, category in enumerate(CATEGORIES):
                st.write(f'{category} : {probabilities[index]*100:.2f}%')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
