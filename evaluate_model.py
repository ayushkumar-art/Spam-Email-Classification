import joblib
from sklearn.metrics import classification_report
from src.train_model import train

def evaluate():
    model, X_test, y_test, _ = train()
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    evaluate()
