import os
import streamlit as st
import requests
from dotenv import load_dotenv

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()

API_URL = os.getenv("API_URL")

if not API_URL:
    st.error("❌ API_URL not found. Please set it in the .env file.")
    st.stop()
#API_URL = "http://api:8000/predict"  # Docker service name
# For local testing without Docker, use:
#API_URL = "http://localhost:8000/predict"

# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🧠 Sentiment Analysis")
st.write("Enter text below to analyze sentiment using a fine-tuned BERT model.")

# -----------------------------
# User Input
# -----------------------------
user_text = st.text_area("Input Text", height=150)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"text": user_text},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.success(f"Sentiment: **{sentiment}**")
                st.info(f"Confidence Score: **{confidence}**")
            else:
                st.error("Failed to get prediction from API.")

        except requests.exceptions.RequestException as e:
            st.error("API is not reachable. Please ensure the backend is running.")
