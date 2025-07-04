
import streamlit as st
import pickle

# --- Load models and preprocessing tools ---
with open("logistic_model.pkl", "rb") as f:
    lr_model = pickle.load(f)  # lr_model used in training

with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)  # nb_model used in training

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Streamlit UI ---
st.title("📝 Sentiment Analysis App")
st.markdown("Enter a product review and choose a model to classify it as **Positive**, **Negative**, or **Neutral**.")

# Text input
review = st.text_area("✏️ Enter your review:")

# Model selector
model_choice = st.radio("🔍 Choose model:", ["Logistic Regression", "Naive Bayes"])

# Predict button
if st.button("Predict"):
    if not review.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        # Transform input
        X_input = tfidf_vectorizer.transform([review])

        # Predict
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(X_input)[0]
        else:
            prediction = nb_model.predict(X_input)[0]

        # Decode output
        sentiment = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Sentiment: **{sentiment.upper()}**")
