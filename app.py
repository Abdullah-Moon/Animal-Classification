import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = load_model("animal_class.keras")
scaler = joblib.load("ac_scalar.pkl")

st.set_page_config(
    page_title="Animal Classification",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Animal Classification")
st.write("Enter the animal characteristics below and click **Predict**.")

# -----------------------------
# Feature Inputs
# -----------------------------
hair = st.selectbox("Hair", [0, 1])
feathers = st.selectbox("Feathers", [0, 1])
eggs = st.selectbox("Eggs", [0, 1])
milk = st.selectbox("Milk", [0, 1])
airborne = st.selectbox("Airborne", [0, 1])
aquatic = st.selectbox("Aquatic", [0, 1])
predator = st.selectbox("Predator", [0, 1])
toothed = st.selectbox("Toothed", [0, 1])
backbone = st.selectbox("Backbone", [0, 1])
breathes = st.selectbox("Breathes", [0, 1])
venomous = st.selectbox("Venomous", [0, 1])
fins = st.selectbox("Fins", [0, 1])

legs = st.slider("Legs", 0, 8, 4)

tail = st.selectbox("Tail", [0, 1])
domestic = st.selectbox("Domestic", [0, 1])
catsize = st.selectbox("Cat Size", [0, 1])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Animal Class"):

    features = np.array([[
        hair,
        feathers,
        eggs,
        milk,
        airborne,
        aquatic,
        predator,
        toothed,
        backbone,
        breathes,
        venomous,
        fins,
        legs,
        tail,
        domestic,
        catsize
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    predicted_class = np.argmax(prediction) + 1

    class_names = {
        1: "Mammal 🐶",
        2: "Bird 🦅",
        3: "Reptile 🦎",
        4: "Fish 🐟",
        5: "Amphibian 🐸",
        6: "Bug 🐞",
        7: "Invertebrate 🐙"
    }

    st.success(f"Predicted Class: {class_names[predicted_class]}")

    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i+1]} : {prob:.2%}")