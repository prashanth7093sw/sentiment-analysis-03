import streamlit as st
import pickle
import numpy as np

# Load saved models and encoders
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("logistic_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Streamlit UI
st.title("Sentiment Analysis App (Logistic Regression & Naive Bayes)")
st.markdown("Enter a review and see predictions from both models.")

user_input = st.text_area("Enter Customer Review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform text using TF-IDF
        X_input = tfidf.transform([user_input])

        # Predict using Logistic Regression
        lr_pred = lr_model.predict(X_input)
        lr_proba = lr_model.predict_proba(X_input)

        # Predict using Naive Bayes
        nb_pred = nb_model.predict(X_input)
        nb_proba = nb_model.predict_proba(X_input)

        # Decode predictions
        lr_sentiment = label_encoder.inverse_transform(lr_pred)[0]
        nb_sentiment = label_encoder.inverse_transform(nb_pred)[0]

        st.subheader("🔍 Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Logistic Regression")
            st.write(f"**Predicted Sentiment:** {lr_sentiment}")
            for idx, label in enumerate(label_encoder.classes_):
                st.write(f"{label}: {lr_proba[0][idx]:.4f}")

        with col2:
            st.markdown("### Naive Bayes")
            st.write(f"**Predicted Sentiment:** {nb_sentiment}")
            for idx, label in enumerate(label_encoder.classes_):
                st.write(f"{label}: {nb_proba[0][idx]:.4f}")

