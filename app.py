import streamlit as st
import joblib, nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import linkedin_data

# Download necessary resources for NLTK if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

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
# Function to predict sentiment and provide interpretation
def predict_sentiment(text, model, vectorizer):
    # Preprocess the text
    text = preprocess_text(text)
    processed_text = ' '.join(text)  # Join tokens into a single string

    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    print(model.predict(text_vectorized))

    # Get the probabilities for each class
    probabilities = model.predict_proba(text_vectorized)[0]
    print("Probability of sentiment:", probabilities)

    # Inspect the most influential words for positive sentiment
    feature_names = np.array(vectorizer.get_feature_names_out())


    # Interpretation
    if prediction == 1:
        print("The model predicted positive sentiment.")
        # Get the log probabilities of features given the positive class
        feature_log_prob = model.feature_log_prob_[1]
        # Sort the feature log probabilities and get the indices
        top_positive_indices = np.argsort(feature_log_prob)[::-1]
        # Map the indices to the feature names
        top_positive_words = feature_names[top_positive_indices]

        return top_positive_words,"positive", probabilities
    
    elif prediction == -1:
        print("The model predicted negative sentiment.")
        feature_log_prob = model.feature_log_prob_[0]
        # Sort the feature log probabilities and get the indices
        top_negative_indices = np.argsort(feature_log_prob)[::-1]
        # Map the indices to the feature names
        top_negative_words = feature_names[top_negative_indices]

        return top_negative_words,"negative", probabilities

# Load the trained Naive Bayes model and CountVectorizer
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Sentiment Analysis Of News", "LinkedIn Data Server"])

# -------------------------------------------------------------------------------------------------------------------------------------------------
if page == "Sentiment Analysis Of News":
    # Streamlit UI
    st.header("News Article Sentiment Analysis", divider='rainbow')

    user_text = st.text_area("Enter your news here:")

    if st.button("Predict"):
        top_words, sentiment, probabilities = predict_sentiment(user_text, nb_model, vectorizer)
        
        st.write("Sentiment & Probability:")

        if sentiment == "positive":
            st.success(f"The sentiment of News is: {sentiment.capitalize()}")
            st.info(f"Positive: {probabilities[1]}")
        elif sentiment == "negative":
            st.error(f"The sentiment of News is: {sentiment.capitalize()}")
            st.info(f"Negative: {probabilities[0]}")

        st.subheader("We are giving you this sentiment of News because of this sentence:")

        lst = []
        for i in user_text.split():
            if i in top_words:
                lst.append(i)
        sentence = ' '.join(lst)
        st.write(sentence)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
elif page == "LinkedIn Data Server":
    st.header("LinkedIn Data Server")
    # st.header("News Article Sentiment Analysis", divider='rainbow')

    company_name = st.text_input("Enter company name")

    if st.button("Get Details"):

        details = linkedin_data.scrape_company_data(company_name)
        st.write(details)

