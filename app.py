from flask import Flask, jsonify, request 
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from textblob import TextBlob
from flask_cors import CORS


positive_features = [
    "build quality",
    "operating system",
    "performance",
    "security",
    "ecosystem integration",
    "camera quality",
    "customer support",
    "camera"
]

negative_features = [
    "price",
    "customization",
    "closed ecosystem",
    "dependency on itunes",
    "battery life",
    "lack of expandable storage",
    "repairability"
]

app = Flask(__name__)
CORS(app) 


@app.route('/') 
def index():
    return "<center><h1>To use the API put / followed by your text on the root url!</h1></center>"

max_len = 20


def preprocess_tweet(tweet):
    tweet = tweet.lower()  # Convert to lowercase
    tweet = " ".join(word for word in tweet.split() if not word.startswith('@'))  # Remove mentions
    tweet = " ".join(word for word in tweet.split() if not word.startswith('http'))  # Remove URLs
    tweet = " ".join(word for word in tweet.split() if word.isalnum())  # Keep only alphanumeric characters
    return tweet

def classify_tweet_sentiment(tweet):
    # Calculate sentiment polarity using TextBlob
    blob = TextBlob(tweet.lower())
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    flag = 0
    if subjectivity==0.0:
        return "General Fact"
    # Check sentiment polarity for positive features
    for feature in positive_features:
        if feature in tweet.lower() and polarity >= 0:
            flag=1
            return "Real"
    
    # Check sentiment polarity for negative features
    for feature in negative_features:
        if feature in tweet.lower() and polarity < 0:
            flag=1
            return "Real"
    
    # If sentiment polarity doesn't match for any features, classify as fake
    if flag==0:
        return "Entity not found"
    return "May be fake"

@app.route('/<st>', methods=['GET'])
def detect(st):
    input_text = request.args.get('in')
    rs = predict_sentiment(st)
    return jsonify({'prediction': rs})




def predict_sentiment(text):
    text = preprocess_tweet(text)
    prediction = classify_tweet_sentiment(text)
    return prediction



if __name__ == "__main__":
    app.run(debug=False)
