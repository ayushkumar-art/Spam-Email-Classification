# 📧 Spam Email Classifier

A machine learning project that classifies emails as spam or not spam using the UCI Spambase dataset and a Naive Bayes classifier.

## 📁 Project Structure

- `data/`: Raw dataset and metadata
- `notebooks/`: Jupyter notebook for EDA and experimentation
- `src/`: Scripts for preprocessing, training, evaluation, and prediction
- `models/`: Saved ML model and vectorizer
- `outputs/`: Evaluation metrics
- `requirements.txt`: Python dependencies
- `README.md`: Project overview

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/train_model.py
python src/evaluate_model.py
```

## 📊 Model

- Feature extraction: TF-IDF
- Classifier: Multinomial Naive Bayes
