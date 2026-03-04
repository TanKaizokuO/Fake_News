import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocess import preprocess_pipeline
from features import WritingStyleVectorizer

def load_data():
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
    return df.sample(frac=1).reset_index(drop=True)

def train_and_evaluate():
    print("Loading and Preprocessing...")
    df = preprocess_pipeline(load_data())
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[['full_text', 'cleaned_text']], df['label'], test_size=0.2)
    vectorizer = WritingStyleVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)}
    best_model, best_f1 = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} F1: {f1:.4f}")
        if f1 > best_f1: best_f1, best_model = f1, model
    
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("Saved best model.")

if __name__ == "__main__": train_and_evaluate()
