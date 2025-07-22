import joblib

def predict(email_features):
    model = joblib.load("models/spam_classifier_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    email_string = ' '.join(map(str, email_features))
    email_vector = vectorizer.transform([email_string])
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    # Example: 57-feature dummy email vector (replace with real email features for actual use)
    dummy_email = [0]*57
    print(predict(dummy_email))
