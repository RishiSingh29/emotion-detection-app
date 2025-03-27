import streamlit as st
import joblib

# Load the trained model and vectorizer
rf_model = joblib.load("models/random_forest_emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit UI
st.title("😃 Emotion Detection App")
st.write("Enter a sentence to detect the emotion!")

# User input
user_input = st.text_area("Type your text here...")

if st.button("Predict Emotion"):
    if user_input.strip():
        # Transform input text
        text_vectorized = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = rf_model.predict(text_vectorized)[0]
        
        # Display result
        st.subheader(f"Predicted Emotion: **{prediction}**")
    else:
        st.warning("Please enter some text before predicting.")

# Footer
st.write("Made with ❤️ using Streamlit")
