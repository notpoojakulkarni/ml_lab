import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# English text and corresponding labels
data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these cheese", "pos"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can't deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg"),
    ("This is an awesome place", "pos"),
    ("I do not like the taste of this juice", "neg"),
    ("I love to dance", "pos"),
    ("I am sick and tired of this place", "neg"),
    ("What a great holiday", "pos"),
    ("That is a bad locality to stay", "neg"),
    ("We will have good fun tomorrow", "pos"),
    ("I went to my enemy's house today", "neg")
]

# Convert the data to DataFrame
df = pd.DataFrame(data, columns=["Text", "Label"])

# Step 1: Total Instances of Dataset
total_instances = len(df)
print("Total Instances of Dataset:", total_instances)

# Step 2: Preprocess the data and split into training and testing sets
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Text"])
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the classifier
y_pred = nb_classifier.predict(X_test)

# Step 5: Obtain Accuracy, Recall, and Precision values
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label="pos")
precision = precision_score(y_test, y_pred, pos_label="pos")

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)

# Step 6: Draw Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
