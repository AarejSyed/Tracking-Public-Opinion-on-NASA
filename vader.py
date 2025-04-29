import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("posts.csv")

# Drop rows with missing values
df = df.dropna()

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to map VADER compound score to discrete sentiment labels
def map_vader_to_label(score):
    if score > 0.05:  # Positive sentiment threshold
        return 1
    elif score < -0.05:  # Negative sentiment threshold
        return -1
    else:  # Neutral sentiment
        return 0

# Apply VADER to each post and map to sentiment labels
df["vader_predicted"] = df["text"].apply(lambda x: map_vader_to_label(vader_analyzer.polarity_scores(x)["compound"]))

# Evaluate predictions
true_labels = df["label"].tolist()  # Hand-labeled sentiment
predicted_labels = df["vader_predicted"].tolist()

# Print evaluation metrics
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Negative", "Neutral", "Positive"]))

# Save results to a new CSV file
output_df = df[["text", "label", "vader_predicted"]]
output_df.to_csv("vader_predictions.csv", index=False)
print("Predictions saved to vader_predictions.csv")