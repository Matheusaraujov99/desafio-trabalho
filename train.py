import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from utils import normalize_text
import os

DATA_PATH = 'data/examples.csv'
MODEL_DIR = 'model'
VEC_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
CLF_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text', 'label'])
    return df

def preprocess(df):
    df['clean'] = df['text'].astype(str).apply(normalize_text)
    return df

def train():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    X_texts = df['clean'].tolist()
    y = df['label'].tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(X_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    dump(vectorizer, VEC_PATH)
    dump(clf, CLF_PATH)
    print(f"Modelos salvos em {MODEL_DIR}")

if __name__ == '__main__':
    train()