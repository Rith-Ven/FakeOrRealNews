import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression # Different model
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('news.csv')
x_train, x_test, y_train, y_test = train_test_split(df['text'], df.label, test_size = 0.2, random_state = 7)

# Captures 1 and 2 word phrases, ignores phrases appearing less than 3 times, keep only 25,000 most relevant features
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, ngram_range = (1,2), min_df = 3, max_features = 25000)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=100, C=0.5, random_state=7)
pac.fit(tfidf_train, y_train)

#lrModel = LogisticRegression(C = 2.0, max_iter = 1000)
#lrModel.fit(tfidf_train, y_train)

predictions = pac.predict(tfidf_test)
new_score = accuracy_score(y_test, predictions)
print(f"New Accuracy: {round(new_score*100,2)}")

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

with open('model.pkl', 'wb') as model_file:
     pickle.dump(pac,model_file)

print("save success")