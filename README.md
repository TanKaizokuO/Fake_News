# Fake News Detector Using Writing Style

Goal: Classify news articles as **FAKE or REAL** using **writing style features only**.

## Features
- **TF-IDF:** Word frequency patterns (bi-grams).
- **Stylistic:** Average sentence length, punctuation frequency (!, ?, ...), and capitalization ratio.
- **Sentiment:** Polarity and subjectivity scores using TextBlob.
- **Clickbait:** Sensationalist phrase detection.

## Structure
- `src/`: Core logic
  - `preprocess.py`: Text cleaning and cleaning pipeline.
  - `features.py`: Stylistic and TF-IDF feature extraction.
  - `train.py`: Model training and comparison (Logistic Regression, Random Forest).
  - `predict.py`: Inference class.
- `app.py`: Streamlit web interface.
- `models/`: Trained model binaries (gitignored).
- `data/`: Dataset storage (gitignored).

## How to Run

### Using `uv` (Recommended)
1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   uv run python src/train.py
   ```
3. **Launch the Web UI:**
   ```bash
   uv run streamlit run app.py
   ```

### Using standard `pip`
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python src/train.py
   ```
3. **Launch the Web UI:**
   ```bash
   streamlit run app.py
   ```

## Future Upgrades
- **BERT-based classification:** Moving beyond frequency to contextual understanding.
- **Propaganda detection:** Identifying emotional manipulation patterns.
- **RAG-based fact verification:** Adding a layer to check claims against trusted databases.
