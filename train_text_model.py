import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0   # Fake = 0
true["label"] = 1   # Real = 1

# Combine datasets
df = pd.concat([fake, true], axis=0)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Features and labels
X = df["text"]
y = df["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel saved successfully ✅")