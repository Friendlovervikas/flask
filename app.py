import os
import re
import tweepy
import pandas as pd
from textblob import TextBlob
from flask import Flask, render_template, request, redirect, url_for, send_file
from io import StringIO
import tempfile
import nltk
from nltk.corpus import stopwords
from flair.models import TextClassifier
from flair.data import Sentence

nltk.download('stopwords')

app = Flask(__name__)

# ---------------- Twitter API & Flair Setup ----------------
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAM%2FA0AEAAAAAbE%2FeRWj4VLdDKYeTKuPyGwiUQqo%3DZiAt3r1i2Pi2zqskk0K2noQOb1p4KcLCkPIfpEStWG6R3TRZnt'
api = tweepy.Client(bearer_token=BEARER_TOKEN)
classifier = TextClassifier.load('en-sentiment')

# ---------------- Utility Functions ----------------

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"

def preprocess_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    return tweet

def get_polarity_words(tweet):
    analysis = TextBlob(tweet)
    words = analysis.words
    word_polarities = {word: TextBlob(word).sentiment.polarity for word in words}
    sorted_words = sorted(word_polarities.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word, polarity in sorted_words[:5]]  
    return ', '.join(top_words), analysis.sentiment.polarity

def get_tweets_from_csv(file_content):
    try:
        data = pd.read_csv(StringIO(file_content))
        tweets = data.to_dict(orient='records')
        for tweet in tweets:
            tweet['sentiment'] = get_tweet_sentiment(tweet['content'])
            tweet['preprocessed_content'] = preprocess_text(tweet['content'])
            top_words, polarity_score = get_polarity_words(clean_tweet(tweet['content']))
            tweet['top_polarity_words'] = top_words
            tweet['polarity_score'] = polarity_score
        return tweets
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return []

def preprocess_flair(text):
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def analyze_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    if sentence.labels:
        return sentence.labels[0].value
    return "NEUTRAL"

# ---------------- Flask Routes ----------------

@app.route('/')
def home():
    return render_template("features.html")

@app.route("/predict", methods=['POST'])
def pred():
    try:
        csv_file = request.files['csv_file']
        file_content = csv_file.read().decode('utf-8')
        fetched_tweets = get_tweets_from_csv(file_content)
        
        processed_df = pd.DataFrame(fetched_tweets)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        processed_df.to_csv(temp_file.name, index=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        return render_template('result.html', result=fetched_tweets, csv_download=True, temp_file_path=temp_file_path)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/download_csv')
def download_csv():
    temp_file_path = request.args.get('temp_file_path')
    if temp_file_path and os.path.exists(temp_file_path):
        return send_file(temp_file_path, mimetype='text/csv', download_name='processed_tweets.csv', as_attachment=True)
    else:
        return "File not found", 404

@app.route("/predict1", methods=['POST'])
def pred1():
    text = request.form['txt']
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        text_sentiment = "positive"
    elif blob.sentiment.polarity == 0:
        text_sentiment = "neutral"
    else:
        text_sentiment = "negative"
    return render_template('result1.html', msg=text, result=text_sentiment)

@app.route("/results", methods=["POST"])
def results():
    hashtag = request.form['hashtag']
    tweets = []
    results = api.search_recent_tweets(query=f"#{hashtag}", max_results=10)
    if results.data:
        for t in results.data:
            original = t.text
            cleaned = preprocess_flair(original)
            sentiment = analyze_flair(cleaned)
            tweets.append({
                'original': original,
                'cleaned': cleaned,
                'sentiment': sentiment
            })
    return render_template("results.html", tweets=tweets, hashtag=hashtag)

# ---------------- Run Flask ----------------

if __name__ == '__main__':
    # Get the port from the environment variable 'PORT' or default to 5000 if not set
    port = int(os.environ.get('PORT', 5000))
    # Run the app, listening on all public IPs (0.0.0.0) and the specified port
    # Set debug to False for production environments
    app.run(host='0.0.0.0', port=port, debug=False)
