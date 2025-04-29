import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the dataset
df = pd.read_csv("posts.csv")

# Drop rows with missing values
df = df.dropna()

# Remap labels from (-1, 0, 1) to (0, 1, 2)
label_mapping = {-1: 0, 0: 1, 1: 2}
df["mapped_label"] = df["label"].map(label_mapping)

texts = df["text"].tolist()
labels = df["mapped_label"].tolist()

# Split into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,  # Include the original text for alignment
        }

# Datasets and loaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
all_predictions = []
all_labels = []
all_texts = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_texts.extend(batch["text"])  # Collect original texts

# Reverse map predictions and labels to original sentiment values (-1, 0, 1)
reverse_map = {0: -1, 1: 0, 2: 1}
all_predictions = [reverse_map[pred] for pred in all_predictions]
all_labels = [reverse_map[label] for label in all_labels]

# Save predictions to a new CSV file
output_df = pd.DataFrame({
    "text": all_texts,  # Original texts from the test set
    "true_label": all_labels,  # True labels
    "predicted_label": all_predictions  # Model predictions
})
output_df.to_csv("model_predictions.csv", index=False)
print("Predictions saved to model_predictions.csv")

# Report
print("Accuracy:", accuracy_score(all_labels, all_predictions))
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=["Negative", "Neutral", "Positive"]))

# Save model
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")

# Prediction function with reverse mapping
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    reverse_map = {0: -1, 1: 0, 2: 1}
    return sentiment_map[prediction], reverse_map[prediction]

# Example
example_text = "NASA is doing amazing work!"
sentiment, raw_label = predict_sentiment(example_text)
print(f"Prediction: {sentiment} (Label: {raw_label})")