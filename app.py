import streamlit as st
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from predict import FakeNewsPredictor

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.markdown("<style>.stButton>button {width: 100%; background-color: #007bff; color: white;}</style>", unsafe_allow_html=True)

def main():
    st.title("📰 Writing Style Fake News Detector")
    if not os.path.exists('models/best_model.pkl'):
        st.warning("⚠️ Model not found! Run training first.")
        return
    article_text = st.text_area("Paste news text:", height=250)
    if st.button("Analyze Style"):
        if article_text.strip():
            with st.spinner("Analyzing..."):
                predictor = FakeNewsPredictor()
                label, confidence = predictor.predict(article_text)
                color = "#2e7d32" if label == "REAL" else "#c62828"
                st.markdown(f"<div style='padding:20px; border-radius:10px; text-align:center; background-color:{color}22; border:1px solid {color}; color:{color};'><h1>{label}</h1><p>Confidence: {confidence*100:.2f}%</p></div>", unsafe_allow_html=True)
        else: st.error("Please enter text.")

if __name__ == "__main__": main()
