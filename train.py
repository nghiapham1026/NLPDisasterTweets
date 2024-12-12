import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Text preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z#]", " ", text)  # Keep only letters and hashtags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text

# Apply text cleaning
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# Handle missing keywords
train_df['keyword'] = train_df['keyword'].fillna("missing")
test_df['keyword'] = test_df['keyword'].fillna("missing")

# Combine text and keyword as features
train_df['combined_text'] = train_df['keyword'] + " " + train_df['text']
test_df['combined_text'] = test_df['keyword'] + " " + test_df['text']

# Splitting training data into train and validation sets
X = train_df['combined_text']
y = train_df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a TF-IDF vectorizer and Naive Bayes pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Precision: {precision:.2f}")
print(f"Validation Recall: {recall:.2f}")
print(f"Validation F1 Score: {f1:.2f}")

# Make predictions on the test set
test_predictions = pipeline.predict(test_df['combined_text'])

# Prepare submission
test_df['target'] = test_predictions
submission = test_df[['id', 'target']]
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")