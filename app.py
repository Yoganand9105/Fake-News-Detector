import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts
model = tf.keras.models.load_model("artifacts/model.h5")
tokenizer = pickle.load(open("artifacts/tokenizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection System")

text = st.text_area("Paste news article text here")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)
    prob = model.predict(padded)[0][0]

    st.write("### Confidence:", round(float(prob), 3))

    if prob > 0.5:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")
