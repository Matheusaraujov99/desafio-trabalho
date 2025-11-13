import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pdfminer.high_level import extract_text

nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('portuguese'))
STEMMER = SnowballStemmer('portuguese')

def read_pdf(file_path):
    text = extract_text(file_path)
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zà-ú0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return ' '.join(tokens)