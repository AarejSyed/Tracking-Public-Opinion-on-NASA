import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# Load the pretrained model and tokenizer from the folder
model_folder = "roberta_sentiment_model"  # Replace with the path to your saved model folder
tokenizer = RobertaTokenizer.from_pretrained(model_folder)
model = RobertaForSequenceClassification.from_pretrained(model_folder)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Load the datasets
before_df = pd.read_csv("before.csv")
after_df = pd.read_csv("after.csv")

# Drop rows with missing values
before_df = before_df.dropna(subset=["text", "score"])
after_df = after_df.dropna(subset=["text", "score"])

# Map labels to integers
label_mapping = {-1: 0, 0: 1, 1: 2}
reverse_mapping = {0: -1, 1: 0, 2: 1}

before_df["mapped_label"] = before_df["score"].map(label_mapping)
after_df["mapped_label"] = after_df["score"].map(label_mapping)

# Function to predict sentiment
def predict_sentiment(texts):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            predictions.append(prediction)
    return predictions

# Predict and evaluate for "before.csv"
before_texts = before_df["text"].tolist()
before_labels = before_df["mapped_label"].tolist()
before_predictions = predict_sentiment(before_texts)

print("Evaluation for before.csv:")
print("Accuracy:", accuracy_score(before_labels, before_predictions))
print("Classification Report:")
print(classification_report(before_labels, before_predictions, target_names=["Negative", "Neutral", "Positive"]))

# Save results for "before.csv"
before_df["predicted_label"] = before_predictions
before_df["predicted_score"] = before_df["predicted_label"].map(reverse_mapping)
before_df.to_csv("before_results.csv", index=False)

# Predict and evaluate for "after.csv"
after_texts = after_df["text"].tolist()
after_labels = after_df["mapped_label"].tolist()
after_predictions = predict_sentiment(after_texts)

print("\nEvaluation for after.csv:")
print("Accuracy:", accuracy_score(after_labels, after_predictions))
print("Classification Report:")
print(classification_report(after_labels, after_predictions, target_names=["Negative", "Neutral", "Positive"]))

# Save results for "after.csv"
after_df["predicted_label"] = after_predictions
after_df["predicted_score"] = after_df["predicted_label"].map(reverse_mapping)
after_df.to_csv("after_results.csv", index=False)