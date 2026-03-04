import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(df, text_column='text'):
    if 'title' in df.columns:
        df['full_text'] = df['title'] + " " + df[text_column]
    else:
        df['full_text'] = df[text_column]
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    stop_words = set(stopwords.words('english'))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    return df
