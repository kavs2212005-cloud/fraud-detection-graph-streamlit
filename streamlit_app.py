import streamlit as st
import joblib
import pickle
import numpy as np

st.title("🔐 Credit Card Fraud Detection Using Graph Embeddings")
st.write("This app predicts the probability of a transaction being fraudulent using a graph-based AI model.")

# -----------------------------
# Load Model + Scaler + Embeddings
# -----------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("xgb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open("node_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return model, scaler, embeddings

model, scaler, embeddings = load_model_files()

# get embedding safely
def get_embedding(node_id):
    if node_id in embeddings:
        return embeddings[node_id]
    # return zero vector if node not found
    return np.zeros(len(next(iter(embeddings.values()))))

# -----------------------------
# Input Section
# -----------------------------
st.header("Enter Transaction Details")

user_id = st.text_input("User ID", "user_1")
card_id = st.text_input("Card ID", "card_1")
merchant_id = st.text_input("Merchant ID", "merchant_1")
device_id = st.text_input("Device ID", "device_1")

amount = st.number_input("Transaction Amount", value=50.0)
hour = st.number_input("Hour of Transaction (0–23)", min_value=0, max_value=23, value=14)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Fraud Probability"):
    # Get embeddings
    user_vec = get_embedding(f"U_{user_id}")
    card_vec = get_embedding(f"C_{card_id}")
    merchant_vec = get_embedding(f"M_{merchant_id}")
    device_vec = get_embedding(f"D_{device_id}")

    # Combine all features
    feature_vector = np.hstack([
        user_vec,
        card_vec,
        merchant_vec,
        device_vec,
        [amount, hour]
    ])

    # Scale inputs
    feature_scaled = scaler.transform(feature_vector.reshape(1, -1))

    # Predict probability
    fraud_prob = model.predict_proba(feature_scaled)[0][1]

    st.subheader("Fraud Probability:")
    st.write(f"### 🚨 {fraud_prob:.4f}")

    if fraud_prob > 0.7:
        st.error("⚠️ High Fraud Risk!")
    elif fraud_prob > 0.3:
        st.warning("🟠 Moderate Fraud Risk.")
    else:
        st.success("🟢 Low Fraud Risk.")
