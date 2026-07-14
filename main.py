import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# ==============================
# Load trained model and scaler
# ==============================
model = load_model("animal_class.keras")
scaler = joblib.load("ac_scalar.pkl")

# ==============================
# Class Names
# ==============================
class_names = {
    0: "Mammal 🐶",
    1: "Bird 🦅",
    2: "Reptile 🦎",
    3: "Fish 🐟",
    4: "Amphibian 🐸",
    5: "Bug 🐞",
    6: "Invertebrate 🐙"
}

# ==============================
# Streamlit Page
# ==============================
st.set_page_config(
    page_title="Animal Classification",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Animal Classification using Deep Learning")
st.write("Select the animal characteristics below.")

st.divider()

# ==============================
# User Inputs
# ==============================

col1, col2 = st.columns(2)

with col1:
    hair = st.selectbox("Hair", [0, 1])
    feathers = st.selectbox("Feathers", [0, 1])
    eggs = st.selectbox("Eggs", [0, 1])
    milk = st.selectbox("Milk", [0, 1])
    airborne = st.selectbox("Airborne", [0, 1])
    aquatic = st.selectbox("Aquatic", [0, 1])
    predator = st.selectbox("Predator", [0, 1])
    toothed = st.selectbox("Toothed", [0, 1])

with col2:
    backbone = st.selectbox("Backbone", [0, 1])
    breathes = st.selectbox("Breathes", [0, 1])
    venomous = st.selectbox("Venomous", [0, 1])
    fins = st.selectbox("Fins", [0, 1])
    legs = st.slider("Number of Legs", 0, 8, 4)
    tail = st.selectbox("Tail", [0, 1])
    domestic = st.selectbox("Domestic", [0, 1])
    catsize = st.selectbox("Cat Size", [0, 1])

# ==============================
# Prediction
# ==============================

if st.button("Predict Animal", use_container_width=True):

    data = np.array([[
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

    # Scale input
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = prediction[0][predicted_class] * 100

    st.success(
        f"### Prediction: {class_names[predicted_class]}"
    )

    st.info(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.progress(float(prob))
        st.write(f"{class_names[i]} : {prob*100:.2f}%")