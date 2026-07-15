import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv('news.csv')
x_train, x_test, y_train, y_test = train_test_split(df['text'], df.label, test_size = 0.2, random_state = 7)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

with open('model.pkl', 'wb') as model_file:
     pickle.dump(pac,model_file)

print("save success")