# 📧 Spam Email Classifier

This project is a **machine learning-based spam email classifier** built using the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase). It classifies emails as either **spam** or **not spam** based on textual content and metadata features.

---

## 🚀 Features

- Uses **57 extracted features** including word frequencies, character frequencies, and capital run length statistics
- Trained using classic ML algorithms (e.g., Naive Bayes)
- Includes **modular Python scripts** for preprocessing, training, evaluation, and prediction
- Ready for integration with web or API-based front ends
- Includes a **Jupyter Notebook** for interactive exploration and visualization

---

## 🧠 Model Pipeline

1. **Preprocessing**
   - Load and split dataset
   - Normalize and vectorize features
2. **Model Training**
   - Train using Multinomial Naive Bayes (or others)
   - Save model with `joblib`
3. **Evaluation**
   - Evaluate on test set
   - Output precision, recall, F1-score
4. **Prediction**
   - Load model and predict on new input data

---

## 📁 Project Structure

```
Spam-Email-Classifier/
│
├── data/
│   ├── spambase.csv                # Cleaned dataset
│   ├── spambase.names              # Feature names
│   └── spambase.DOCUMENTATION      # Dataset description
│
├── notebooks/
│   └── Spam_Email_Classifier.ipynb # EDA + training notebook
│
├── models/
│   └── spam_classifier_model.pkl   # Trained ML model (generated after training)
│
├── src/
│   ├── data_preprocessing.py       # Preprocessing pipeline
│   ├── train_model.py              # Model training script
│   ├── evaluate_model.py           # Evaluation script
│   └── predict.py                  # Prediction script
│
├── outputs/
│   └── classification_report.txt   # Model evaluation results
│
├── requirements.txt                # Dependencies list
├── .gitignore                      # Ignore unnecessary files
└── README.md                       # You're reading it!
```

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/ayushkumar-art/Spam-Email-Classifier.git
cd Spam-Email-Classifier

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🧪 How to Use

### 🔹 Train the Model

```bash
python src/train_model.py
```

### 🔹 Evaluate the Model

```bash
python src/evaluate_model.py
```

### 🔹 Predict New Email

```bash
python src/predict.py
```

Or, run everything interactively inside:

```
notebooks/Spam_Email_Classifier.ipynb
```

---

## 📊 Evaluation Metrics (Example)

| Metric     | Score  |
|------------|--------|
| Accuracy   | 94.5%  |
| Precision  | 96.0%  |
| Recall     | 92.3%  |
| F1-score   | 94.1%  |

*(Varies slightly based on train/test split)*

---

## 📚 Dataset

- **Source:** [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Samples:** 4,601 emails
- **Features:** 57 numeric + 1 label (`spam = 1`, `not spam = 0`)

---

## 👨‍💻 Author

Ayush Kumar  
Feel free to connect with me on LinkedIn-www.linkedin.com/in/ayush-kumar-866b99259.
