import joblib
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_pipeline
from features import WritingStyleVectorizer

class FakeNewsPredictor:
    def __init__(self, model_path='models/best_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    def predict(self, text):
        df = preprocess_pipeline(pd.DataFrame({'text': [text]}))
        X = self.vectorizer.transform(df)
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0][prediction] if hasattr(self.model, "predict_proba") else 1.0
        return ("REAL" if prediction == 1 else "FAKE", confidence)
