import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset (assuming posts.csv is already created)
df = pd.read_csv("posts.csv")

# Preprocess the data
df = df.dropna()  # Drop rows with missing values
texts = df["text"].tolist()
labels = df["label"].tolist()  # Use the hand-labeled column `label` (1, 0, -1)

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Use unigrams and bigrams
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_labels)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(test_labels, predictions))
print("Classification Report:")
print(classification_report(test_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))

# Save the model and vectorizer
joblib.dump(model, "tfidf_sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save predictions to a CSV file
output_df = pd.DataFrame({
    "text": test_texts,  # Original texts from the test set
    "true_label": test_labels,  # True labels
    "predicted_label": predictions  # Model predictions
})
output_df.to_csv("tfidf_predictions.csv", index=False)
print("Predictions saved to tfidf_predictions.csv")

# Prediction function
def predict_sentiment(text):
    # Load the saved model and vectorizer
    model = joblib.load("tfidf_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    # Transform the input text
    text_features = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_features)[0]
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment_map[prediction]

# Example prediction
example_text = "NASA is doing amazing work!"
print("Prediction:", predict_sentiment(example_text))