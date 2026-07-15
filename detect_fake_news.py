import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Libraries imported successfully\n")

#---Reads the data 
df = pd.read_csv('news.csv')
#Gets the shape (dimensions) of the data and looks at the head (first five rows)
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 records:")
print(df.head())

#Extract labels to determine if REAL or FAKE
labels = df.label
print("\nTarget labels preview")
print(labels.head())

# ---Split the dataset - 80% training, 20% testing
# df['text'] contains actual text of the articles, x
# labels contains whether the articles are REAL or FAKE, y
# test_size tells us how much of the data is used for testing
# random_state shuffles the data at some point (seed), ensuring that every time it is run it is shuffled the same way
x_train, x_test, y_train, y_test = train_test_split( df['text'], labels, test_size = 0.2, random_state = 7)

# ---Vectorize the text
# A TfidfVectorizer calculates the frequency of words in a document, filtering out common words (like, the, and...) and discard terms in more than 70% of text
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# Fit and transform train set, transform test set
# Fit finds all the unique words and calculates how often they appear (only for training data!)
# Transform converts sentences into a matrix with the TF-IDF score

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# ---Train the Passive-Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

# --Predict and Evaluate Model Performance
y_pred = pac.predict(tfidf_test) 
score = accuracy_score(y_test,y_pred)
print(f'\nAccuracy: {round(score*100,2)}%')

# Build a confusion matrix to see T/F positives and negatives
matrix = confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
print('\nConfusion Matrix:')
print(matrix)