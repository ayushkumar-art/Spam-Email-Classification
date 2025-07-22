import joblib
from sklearn.naive_bayes import MultinomialNB
from src.data_preprocessing import load_data, preprocess_text_data, split_data

def train():
    X, y = load_data("data/spambase.data")
    X_tfidf, vectorizer = preprocess_text_data(X)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/spam_classifier_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    return model, X_test, y_test, y

if __name__ == "__main__":
    train()
