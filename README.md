# News Category Classification using NLP

A machine learning project that classifies news articles into **World, Sports, Business, and Sci/Tech** using Natural Language Processing techniques.

---

## Features

- Text preprocessing (cleaning, stopword removal)
- TF-IDF vectorization
- Multiple ML models:
  - Naive Bayes
  - Support Vector Machine (LinearSVC)
  - Logistic Regression
- Model comparison (Accuracy & F1 Score)
- Streamlit web application interface

---

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Streamlit

---

## Dataset

- **AG News Classification Dataset (Kaggle)**
- 4 Classes:
  - World
  - Sports
  - Business
  - Sci/Tech
- Dataset split:
  - Training set
  - Test set
- Input features:
  - Title + Description

---

## Methodology

### 1. Text Preprocessing
- Lowercasing
- Removing special characters
- Stopword removal

### 2. Feature Extraction
- TF-IDF Vectorization

### 3. Model Training
- Naive Bayes
- Support Vector Machine (LinearSVC)
- Logistic Regression

### 4. Evaluation
- Accuracy
- F1 Score

---

## Results

| Model                | Accuracy | F1 Score |
|---------------------|----------|----------|
| Naive Bayes         | 89.51%   | 0.8947   |
| SVM (LinearSVC)     | 91.51%   | 0.9150   |
| Logistic Regression | **91.64%** | **0.9163** |

---

## Key Insights

- Logistic Regression achieved the best performance
- SVM also performed strongly on high-dimensional text data
- TF-IDF significantly improved feature representation
- Combining Title + Description improved accuracy

---

## Web Application (Streamlit)

### 📰 Main UI
![App Screenshot](https://github.com/shubhankar05sarkar/news-category-classifier/blob/c578c99b69d37f9af3b858e590d24f107516b8fb/main-ui.png)

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Train models

```bash
python main.py
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## Author

Created with ❤️ by **Shubhankar Sarkar**
[GitHub Profile](https://github.com/shubhankar05sarkar)
