pip install nltk scikit-learn pyttsx3

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

common_phrases = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! How can I help you?",
    "thanks": "You're welcome!",
    # More phrases...
}

faq_data = {
    "What is your return policy?": "Our return policy lasts 30 days...",
    # More FAQs...
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return tokens

preprocessed_faq_data = {question: preprocess_text(question) for question in faq_data.keys()}

vectorizer = TfidfVectorizer()

def find_best_match(query, faq_data):
    query_tokens = preprocess_text(query)
    all_texts = [' '.join(preprocessed_faq_data[question]) for question in faq_data.keys()]
    query_text = ' '.join(query_tokens)
    tfidf_matrix = vectorizer.fit_transform(all_texts + [query_text])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_index = cosine_similarities.argmax()
    return list(faq_data.keys())[best_match_index]

def chatbot_response(user_query):
    best_match_question = find_best_match(user_query, faq_data)
    return faq_data[best_match_question]

def chat():
    print("Hello! I'm an FAQ bot. Ask me a question.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        if user_query.lower() in common_phrases:
            print("Chatbot: ", common_phrases[user_query.lower()])
        else:
            response = chatbot_response(user_query)
            print("Chatbot: ", response)
            text_to_speech(response)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

chat()
