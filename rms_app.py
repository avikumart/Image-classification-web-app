import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# base model for feature map prediction of image input
conv_base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                        include_top=False,
                                        input_shape=(150,150,3)
                                        )

# final prediction using our trained model
model = tf.keras.models.load_model("image_classifier_cnn2.h5")

st.write("""
        # Natural scenaries image classification and label prediction
        """)

st.write("This is the most basic image classification web-app to predict the class of images"
        )

file = st.file_uploader("Please upload a natural image file", type=['jpg','png'])


def import_and_predict(image_input, conv_base, model):
    '''
    function to take input file of image and pre-process before feeding 
    to the model to predict the output.
    '''
    size = (150,150)
    image = ImageOps.fit(image_input, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    img = np.expand_dims(image, axis=0)
    img /= 255.
    
    # prediction output of the base model
    
    pre1 = conv_base.predict(img)
    prediction = model.predict(pre1)
    
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, conv_base, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a building image class")
    elif np.argmax(prediction) == 1:
        st.write("It is a forest image class")
    elif np.argmax(prediction) == 2:
        st.write("It is a glacier image class")    
    elif np.argmax(prediction) == 3:
        st.write("It is a mountain image class")
    elif np.argmax(prediction) == 4:
        st.write("It is a sea image class")
    else:
        st.write("It is a street image class")
    
    prediction_probability = pd.DataFrame({"Classes":['building','forest','glacier','mountain','sea','street'],
                                        "Probability": prediction.ravel()})
    
    st.text("Prediction (0:building, 1:forest, 2:glacier, 3:mountain, 4:sea, 5:street)")
    st.write("Prediction_probability of an image")
    st.write(prediction_probability)   
