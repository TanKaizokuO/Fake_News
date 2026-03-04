import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_stylistic_features(text):
    if not isinstance(text, str) or len(text) == 0:
        return {'avg_sentence_length': 0, 'punc_freq_excl': 0, 'punc_freq_ques': 0, 'punc_freq_ellipsis': 0, 'caps_ratio': 0, 'sentiment_polarity': 0, 'sentiment_subjectivity': 0, 'clickbait_score': 0}
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    punc_freq_excl = text.count('!') / len(text)
    punc_freq_ques = text.count('?') / len(text)
    punc_freq_ellipsis = text.count('...') / len(text)
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = len(caps_words) / len(words) if words else 0
    analysis = TextBlob(text)
    clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 'literally', 'amazing', 'must see', 'gone viral']
    clickbait_score = sum(1 for word in clickbait_words if word in text.lower())
    return {
        'avg_sentence_length': avg_sentence_length,
        'punc_freq_excl': punc_freq_excl,
        'punc_freq_ques': punc_freq_ques,
        'punc_freq_ellipsis': punc_freq_ellipsis,
        'caps_ratio': caps_ratio,
        'sentiment_polarity': analysis.sentiment.polarity,
        'sentiment_subjectivity': analysis.sentiment.subjectivity,
        'clickbait_score': clickbait_score
    }

class WritingStyleVectorizer:
    def __init__(self, max_features=5000):
        self.tfidf = TfidfVectorizer(max_features=max_features)
    def fit(self, df):
        self.tfidf.fit(df['cleaned_text'])
        return self
    def transform(self, df):
        tfidf_features = self.tfidf.transform(df['cleaned_text']).toarray()
        features = df['full_text'].apply(extract_stylistic_features)
        style_features = pd.DataFrame(features.tolist()).values
        return np.hstack((tfidf_features, style_features))
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
