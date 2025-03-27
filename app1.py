from flask import Flask, render_template, request
import joblib

# Load trained model and vectorizer
rf_model = joblib.load("models\\random_forest_emotion_model.pkl")
vectorizer = joblib.load("models\\tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_text = request.form['text']  # Get text input from user
        text_vectorized = vectorizer.transform([user_text])  # Transform text
        predicted_emotion = rf_model.predict(text_vectorized)[0]  # Predict emotion
        return render_template('main.html', text=user_text, emotion=predicted_emotion)
    return render_template('main.html', text=None, emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
