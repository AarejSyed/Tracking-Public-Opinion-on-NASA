import pandas as pd
import numpy as np

# Load the results
before_results = pd.read_csv("before_results.csv")
after_results = pd.read_csv("after_results.csv")

# Combine the datasets for overall statistics
combined_results = pd.concat([before_results, after_results], ignore_index=True)

# Calculate percentages of positive, negative, and neutral labels
def calculate_label_percentages(df, column_name):
    total = len(df)
    positive = len(df[df[column_name] == 1]) / total * 100
    negative = len(df[df[column_name] == -1]) / total * 100
    neutral = len(df[df[column_name] == 0]) / total * 100
    return positive, negative, neutral

# Function to calculate accuracy
def calculate_accuracy(df):
    correct_predictions = df[df["score"] == df["predicted_score"]]
    accuracy_fraction = len(correct_predictions) / len(df)
    accuracy_percentage = accuracy_fraction * 100
    return accuracy_fraction, accuracy_percentage

# Function to calculate mean and standard deviation
def calculate_mean_std(df, column_name):
    mean = df[column_name].mean()
    std = df[column_name].std()
    return mean, std

# Calculate statistics for "before_results"
before_truth_positive, before_truth_negative, before_truth_neutral = calculate_label_percentages(before_results, "score")
before_pred_positive, before_pred_negative, before_pred_neutral = calculate_label_percentages(before_results, "predicted_score")
before_truth_mean, before_truth_std = calculate_mean_std(before_results, "score")
before_pred_mean, before_pred_std = calculate_mean_std(before_results, "predicted_score")
before_accuracy_fraction, before_accuracy_percentage = calculate_accuracy(before_results)

# Calculate statistics for "after_results"
after_truth_positive, after_truth_negative, after_truth_neutral = calculate_label_percentages(after_results, "score")
after_pred_positive, after_pred_negative, after_pred_neutral = calculate_label_percentages(after_results, "predicted_score")
after_truth_mean, after_truth_std = calculate_mean_std(after_results, "score")
after_pred_mean, after_pred_std = calculate_mean_std(after_results, "predicted_score")
after_accuracy_fraction, after_accuracy_percentage = calculate_accuracy(after_results)

# Calculate statistics for "combined_results"
combined_truth_positive, combined_truth_negative, combined_truth_neutral = calculate_label_percentages(combined_results, "score")
combined_pred_positive, combined_pred_negative, combined_pred_neutral = calculate_label_percentages(combined_results, "predicted_score")
combined_truth_mean, combined_truth_std = calculate_mean_std(combined_results, "score")
combined_pred_mean, combined_pred_std = calculate_mean_std(combined_results, "predicted_score")
combined_accuracy_fraction, combined_accuracy_percentage = calculate_accuracy(combined_results)

# Print the statistics
print("=== Before Results ===")
print(f"Truth Labels - Positive: {before_truth_positive:.2f}%, Negative: {before_truth_negative:.2f}%, Neutral: {before_truth_neutral:.2f}%")
print(f"Predicted Labels - Positive: {before_pred_positive:.2f}%, Negative: {before_pred_negative:.2f}%, Neutral: {before_pred_neutral:.2f}%")
print(f"Truth Labels - Mean: {before_truth_mean:.2f}, Standard Deviation: {before_truth_std:.2f}")
print(f"Predicted Labels - Mean: {before_pred_mean:.2f}, Standard Deviation: {before_pred_std:.2f}")
print(f"Accuracy: {before_accuracy_percentage:.2f}%")

print("\n=== After Results ===")
print(f"Truth Labels - Positive: {after_truth_positive:.2f}%, Negative: {after_truth_negative:.2f}%, Neutral: {after_truth_neutral:.2f}%")
print(f"Predicted Labels - Positive: {after_pred_positive:.2f}%, Negative: {after_pred_negative:.2f}%, Neutral: {after_pred_neutral:.2f}%")
print(f"Truth Labels - Mean: {after_truth_mean:.2f}, Standard Deviation: {after_truth_std:.2f}")
print(f"Predicted Labels - Mean: {after_pred_mean:.2f}, Standard Deviation: {after_pred_std:.2f}")
print(f"Accuracy: {after_accuracy_percentage:.2f}%")

print("\n=== Combined Results ===")
print(f"Truth Labels - Positive: {combined_truth_positive:.2f}%, Negative: {combined_truth_negative:.2f}%, Neutral: {combined_truth_neutral:.2f}%")
print(f"Predicted Labels - Positive: {combined_pred_positive:.2f}%, Negative: {combined_pred_negative:.2f}%, Neutral: {combined_pred_neutral:.2f}%")
print(f"Truth Labels - Mean: {combined_truth_mean:.2f}, Standard Deviation: {combined_truth_std:.2f}")
print(f"Predicted Labels - Mean: {combined_pred_mean:.2f}, Standard Deviation: {combined_pred_std:.2f}")
print(f"Accuracy: {combined_accuracy_percentage:.2f}%")