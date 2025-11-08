import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_digit_model.h5")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the box below and click **Predict**.")

# Canvas settings
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Convert to image
        img = Image.fromarray((canvas.image_data[:, :, 3] * 255).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")

        # Preprocess
        img_arr = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Predict
        prediction = model.predict(img_arr)
        digit = np.argmax(prediction)

        st.success(f"Predicted Digit: **{digit}**")
