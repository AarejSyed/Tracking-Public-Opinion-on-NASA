from atproto import Client
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Log in to Bluesky
client = Client()
profile = client.login('nasaenthusiast.bsky.social', 'NASANLPPROJECT')

# Store posts
posts = {
    "text": [],
    "date": [],
    "vader_sentiment": [],
    "textblob_sentiment": [],
}

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (including @ and #)
    text = re.sub(r'[@#]\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\d.,!?]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

# Function to map sentiment scores to discrete ratings
def map_sentiment_to_rating(score):
    if score > 0.05:  # Positive sentiment threshold
        return 1
    elif score < -0.05:  # Negative sentiment threshold
        return -1
    else:  # Neutral sentiment
        return 0

# Get sentiment scores using different analyzers
def get_sentiment_scores(text):
    # VADER sentiment
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    vader_rating = map_sentiment_to_rating(vader_compound)
    
    # TextBlob sentiment
    textblob_sentiment = TextBlob(text).sentiment.polarity
    textblob_rating = map_sentiment_to_rating(textblob_sentiment)
    
    return {
        'vader': vader_rating,
        'textblob': textblob_rating
    }

parameters = {
    "q": "NASA",
    "limit": 100,
    "lang": "en",
    "since": "2023-06-28T00:00:00Z",
    "until": "2024-06-28T23:59:59Z",
}

# Get posts
fetch = client.app.bsky.feed.search_posts(parameters)
print(f"Posts fetched: {len(fetch.posts)}")

for post in fetch.posts:
    text = post.record.text
    processed_text = preprocess_text(text)
    sentiment_scores = get_sentiment_scores(processed_text)
    
    posts["text"].append(processed_text)
    posts["date"].append(post.record.created_at)
    posts["vader_sentiment"].append(sentiment_scores['vader'])
    posts["textblob_sentiment"].append(sentiment_scores['textblob'])

# Save posts to CSV
df = pd.DataFrame(posts)
df.to_csv("posts.csv", index=True)
print("Posts saved to posts.csv")