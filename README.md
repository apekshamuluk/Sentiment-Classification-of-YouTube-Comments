# 📊 Sentiment Analysis of YouTube Comments using NLP

This project performs **sentiment analysis on YouTube comments** using **Natural Language Processing (NLP)** techniques.  
The goal is to understand audience opinions by classifying comments into **Positive**, **Negative**, and **Neutral** sentiments.

---

## 🎯 Project Objective
- Analyze YouTube comments
- Clean and preprocess text data
- Convert text into numerical features using **TF-IDF**
- Handle class imbalance
- Train a machine learning model to predict sentiment
- Evaluate model performance
- Predict sentiment for new/unseen comments

---

## 🧠 Technologies Used
- **Python**
- **Pandas, NumPy**
- **NLTK** (Text preprocessing)
- **TextBlob** (Sentiment polarity)
- **Scikit-learn**
- **Matplotlib, Seaborn**

---

## 📂 Dataset
- Source: YouTube comments dataset  
- Total comments after cleaning: **18,364**
- Columns:
  - `Comment` – Original YouTube comment
  - `Sentiment` – Given label
  - `clean_comment` – Preprocessed text
  - `sentiment` – Final sentiment label

### Sentiment Distribution:
- Positive: 11,797  
- Neutral: 4,255  
- Negative: 2,312  

➡ The dataset is **imbalanced**, with more positive comments.

---

## 🧹 Data Preprocessing
The following steps were applied:
- Lowercasing text
- Removing URLs and punctuation
- Tokenization
- Stopword removal
- Lemmatization

This ensures cleaner and more meaningful text for modeling.

---

## 🔢 Feature Extraction
- **TF-IDF Vectorization** was used to convert text into numerical features.
- Each word becomes a feature based on its importance across all comments.

---

## ⚖️ Handling Class Imbalance
Since the dataset is imbalanced, **class weights** were applied in **Logistic Regression** after TF-IDF vectorization.  
This ensures equal importance for all sentiment classes **without modifying the original data distribution**.

---

## 🤖 Model Used
- **Logistic Regression**
- Reason:
  - Works well with TF-IDF features
  - Fast and interpretable
  - Suitable for text classification tasks

---

## 📈 Model Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Accuracy achieved:** ~85%

The confusion matrix visualization helps understand predictions across all sentiment classes.

---

## 🔍 Prediction on New Comments
The trained model can predict sentiment for new YouTube comments, making it useful for real-world applications.

Example:
```python
predict_sentiment("This video is amazing 🔥")
