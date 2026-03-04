# Fake News Detector Using Writing Style

Goal: Classify news articles as **FAKE or REAL** using **writing style features only**.

## Features
- **TF-IDF:** Word frequency.
- **Stylistic:** Sentence length, punctuation, capitalization.
- **Sentiment:** Polarity and subjectivity.
- **Clickbait:** Sensationalist phrase detection.

## Structure
- `src/`: Core logic (preprocess, features, train, predict).
- `app.py`: Streamlit interface.
- `models/`: Trained binaries (gitignored).
- `data/`: Dataset storage (gitignored).

## Usage
1. `pip install -r requirements.txt`
2. Run `python src/train.py`
3. Launch `streamlit run app.py`

## Future Upgrades
- BERT-based classification.
- Propaganda detection.
- RAG-based fact verification.
