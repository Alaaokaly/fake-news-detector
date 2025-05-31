import joblib
import numpy as np
import pandas as pd
import re
import logging
from typing import Tuple, Dict, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def load_model_and_vectorizer(model_name: str, vectorizer_name: str):
    model_path = f"model/{model_name}_model_{vectorizer_name}.pkl"
    vectorizer_path = f"model/{vectorizer_name}_vectorizer.pkl"
    clf = joblib.load(model_path)
    vec = joblib.load(vectorizer_path)
    return clf, vec

def validate_input( text: str, title: str = "") -> Tuple[bool, str]:
        """
        Validate and prepare input text
        """
        if not text or not isinstance(text, str):
            return False, "Text cannot be empty"
        
        # Combine title and text
        combined_text = f"{title} {text}".strip() if title else text
        cleaned_text = clean_text(combined_text)
        
        if len(cleaned_text) < 10:
            return False, "Text is too short (minimum 10 characters after cleaning)"
        
        if len(cleaned_text.split()) < 3:
            return False, "Text must contain at least 3 words"
        
        if len(cleaned_text) < 10:
            return False, "Text is too short (minimum 10 characters after cleaning)"
        
        if len(cleaned_text.split()) < 3:
            return False, "Text must contain at least 3 words"
        
        return True, cleaned_text

def predict(text: str, model_name="lr", vectorizer_name="tfidf"):
    clf, vec = load_model_and_vectorizer(model_name, vectorizer_name)
    _ , cleaned_text = validate_input(text)
    vectorized = vec.transform([cleaned_text])
    pred = clf.predict(vectorized)[0]
    proba = clf.predict_proba(vectorized)[0]
    label_index = list(clf.classes_).index(pred)
    confidence = proba[label_index]
    return pred, confidence



if __name__ == "__main__":
    # Example use
    example_text = "The president said something shocking during the press conference today."
    label, confidence = predict(example_text, model_name="lr", vectorizer_name="tfidf")
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
