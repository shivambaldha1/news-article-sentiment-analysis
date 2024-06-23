from flask import Flask, request, jsonify
import joblib, nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary resources for NLTK if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

# Load the trained Naive Bayes model and CountVectorizer
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess the text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Converting text to lower case
    tokens = [token.lower() for token in tokens]
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# Function to predict sentiment and provide interpretation
def predict_sentiment(text, model, vectorizer):
    # Preprocess the text
    text = preprocess_text(text)
    processed_text = ' '.join(text)  # Join tokens into a single string

    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]

    # Get the probabilities for each class
    probabilities = model.predict_proba(text_vectorized)[0]

    # Inspect the most influential words for positive sentiment
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Interpretation
    if prediction == 1:
        # Get the log probabilities of features given the positive class
        feature_log_prob = model.feature_log_prob_[1]
        # Sort the feature log probabilities and get the indices
        top_positive_indices = np.argsort(feature_log_prob)[::-1]
        # Map the indices to the feature names
        top_positive_words = feature_names[top_positive_indices]
        return top_positive_words, "positive", probabilities
    
    elif prediction == -1:
        feature_log_prob = model.feature_log_prob_[0]
        # Sort the feature log probabilities and get the indices
        top_negative_indices = np.argsort(feature_log_prob)[::-1]
        # Map the indices to the feature names
        top_negative_words = feature_names[top_negative_indices]
        return top_negative_words, "negative", probabilities

@app.route('/', methods=['GET'])
def home():
    return "News Sentiment Analysis API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_text = data['text']
    
    top_words, sentiment, probabilities = predict_sentiment(user_text, nb_model, vectorizer)
    
    response = {
        "sentiment": sentiment,
        "probabilities": {
            "positive": probabilities[1],
            "negative": probabilities[0]
        },
        # "top_words": list(top_words)
    }

    # Highlighting top words in user text
    lst = [i for i in user_text.split() if i in top_words]
    sentence = ' '.join(lst)
    response["highlighted_sentence"] = sentence
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
