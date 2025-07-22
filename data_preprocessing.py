import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def preprocess_text_data(X):
    # Concatenate all columns into a single string to simulate "email text"
    text_data = X.astype(str).agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    X_tfidf = vectorizer.fit_transform(text_data)
    return X_tfidf, vectorizer

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
