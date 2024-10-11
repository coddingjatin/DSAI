from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

nltk.download('punkt')

# Initialize Flaskapp
app = Flask(__name__)

# Load and prepare data
data1 = pd.read_csv('complaints.csv')
data1 = data1.dropna()

# Replace product categories with numerical values
data1["product"] = data1["product"].replace({
    "credit_card": "1",
    "retail_banking": "2",
    "credit_reporting": "3",
    "mortgages_and_loans": "4",
    "debt_collection": "5"
}, regex=False)

# Clean text by removing punctuation
data1['narrative'] = data1['narrative'].str.replace(r'[^\w\s]', '', regex=True)

# Tokenization function
def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    return words

# Split data into features (X) and labels (y)
X = data1['narrative'].values
y = data1["product"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)

# Define category mapping
category_mapping = {
    "1": "Credit Cards",
    "2": "Retail Banking",
    "3": "Credit Reporting",
    "4": "Mortgages and Loans",
    "5": "Debt Collection"
}

# Route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from form
        complaint_text = request.form['complaint']

        # Clean and vectorize the input text
        complaint_cleaned = clean_text(complaint_text)
        complaint_tfidf = tfidf_vectorizer.transform([complaint_cleaned])

        # Make prediction
        prediction = log_reg.predict(complaint_tfidf)
        predicted_category = category_mapping[prediction[0]]

        return render_template('index.html', prediction_text=f'Predicted Category: {predicted_category}')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
