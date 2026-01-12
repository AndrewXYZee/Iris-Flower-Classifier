#Import required libraries
import streamlit as st
import numpy as np
import joblib

#Load model
model = joblib.load('iris_model.joblib')
class_names = ["Setosa", "Versicolor", "Virginica"]

#Setup website
st.set_page_config(page_title="Iris Flower Classifier")
st.title("Iris Flower Classifier")
st.markdown("Setup the measurments of the iris flower and find out its species!")
st.caption("Model accuracy 96.67% on test set")
st.sidebar.header("Flower measurments")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 6.0, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.0,  0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0,  0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.0,  0.1)

#Get input
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#Make prediction
if st.button("Make prediction"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    pred_class = class_names[prediction]

    #Display results
    st.success(f"The flower species is {pred_class}")
    st.caption("Confidence:")
    for name, prob in zip(class_names, probability):
        st.write(f"{name}: {prob:.1%}")

