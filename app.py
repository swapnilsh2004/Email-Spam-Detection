from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        result = 'Spam' if prediction[0] else 'Not Spam'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
