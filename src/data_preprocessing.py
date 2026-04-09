import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='latin-1')
    test_df = pd.read_csv(test_path, encoding='latin-1')

    train_df['text'] = train_df['Title'] + " " + train_df['Description']
    test_df['text'] = test_df['Title'] + " " + test_df['Description']

    train_df['clean_text'] = train_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    return train_df, test_df
