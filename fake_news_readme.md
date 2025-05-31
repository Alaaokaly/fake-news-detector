# 🕵️‍♂️ Fake News Detection System

A machine learning-powered web application that classifies news articles as either **FAKE** or **REAL** using Natural Language Processing techniques and multiple classification algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Performance](#models--performance)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete pipeline for fake news detection, from data preprocessing to model deployment. It uses two different machine learning algorithms (Naive Bayes and Logistic Regression) combined with two text vectorization methods (TF-IDF and Count Vectorizer) to provide accurate classification results.

### Key Components:
- **Data preprocessing and cleaning**
- **Feature extraction using TF-IDF and Count Vectorization**
- **Model training with Naive Bayes and Logistic Regression**
- **Interactive web interface built with Streamlit**
- **Model persistence and evaluation metrics**

## ✨ Features

- 🔍 **Real-time Classification**: Instantly classify news articles as fake or real
- 🎛️ **Multiple Models**: Choose between Naive Bayes and Logistic Regression
- 📊 **Vectorization Options**: TF-IDF or Count Vectorizer for text processing
- 📈 **Confidence Scoring**: Get confidence levels for each prediction
- 🎨 **User-friendly Interface**: Clean, responsive web interface
- 📝 **Comprehensive Logging**: Track all predictions and system events
- 🔧 **Model Evaluation**: Built-in performance metrics and confusion matrices

## 📁 Project Structure

```
fake-news-detection/
├── data/
│   └── fake_or_real_news.csv          # Training dataset
├── model/
│   ├── lr_model_tfidf.pkl             # Logistic Regression + TF-IDF model
│   ├── lr_model_count.pkl             # Logistic Regression + Count model
│   ├── nb_model_tfidf.pkl             # Naive Bayes + TF-IDF model
│   ├── nb_model_count.pkl             # Naive Bayes + Count model
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── count_vectorizer.pkl           # Count vectorizer
│   ├── evaluation.csv                 # Model performance metrics
│   └── predict.py                     # Prediction utilities
├── style.css                          # Streamlit custom styling
├── app.py                             # Main Streamlit application
├── train_model.py                     # Model training script
├── app.log                            # Application logs
└── README.md                          # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
```

4. **Prepare the dataset**
   - Place your `fake_or_real_news.csv` file in the `data/` directory
   - The dataset should contain columns: `text`, `title`, and `label`

5. **Train the models** (if not using pre-trained models)
```bash
python train_model.py
```

## 🎮 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Prediction API

```python
from model.predict import predict

# Make a prediction
text = "Your news article text here..."
prediction, confidence = predict(
    text, 
    model_name="lr",        # "lr" or "nb"
    vectorizer_name="tfidf" # "tfidf" or "count"
)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

### Training New Models

```python
# Run the training script
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train all model combinations
- Save models and vectorizers
- Generate evaluation metrics

## 🧠 Models & Performance

### Available Models

| Model | Vectorizer | Description |
|-------|------------|-------------|
| Logistic Regression | TF-IDF | Linear model with term frequency-inverse document frequency |
| Logistic Regression | Count | Linear model with simple word counts |
| Naive Bayes | TF-IDF | Probabilistic model with TF-IDF features |
| Naive Bayes | Count | Probabilistic model with count features |

### Model Parameters

- **TF-IDF Vectorizer**: 
  - N-gram range: (1,2)
  - Max features: 10,000
  - Min document frequency: 5
  - Max document frequency: 85%

- **Count Vectorizer**:
  - Same parameters as TF-IDF
  - Simple word counting approach

### Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for fake news detection
- **Recall**: Coverage of actual fake news
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 📊 Dataset

The system expects a CSV file with the following structure:

| Column | Description |
|--------|-------------|
| `text` | Main body of the news article |
| `title` | Headline of the article |
| `label` | Classification label ('FAKE' or 'REAL') |

### Data Preprocessing

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove extra whitespaces
   - Remove special characters (keep alphanumeric and periods)
   - Strip leading/trailing spaces

2. **Feature Engineering**:
   - Combine title and text
   - Filter articles with less than 10 characters
   - Remove common stopwords + custom domain-specific stopwords

3. **Validation**:
   - Minimum length requirements
   - Word count validation
   - Input format checking

## 🔧 Technical Details

### Text Processing Pipeline

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()
```

### Custom Stopwords

The system uses enhanced stopword filtering:
- Standard English stopwords (NLTK)
- Custom domain-specific words: 'said', 'will', 'one', 'people', 'now', 'time', 'say', 'reports'

### Model Persistence

All trained models and vectorizers are saved using `joblib` for efficient loading and consistent predictions.

## 📡 API Reference

### Core Functions

#### `predict(text, model_name="lr", vectorizer_name="tfidf")`

**Parameters:**
- `text` (str): Input text to classify
- `model_name` (str): Model choice ("lr" or "nb")
- `vectorizer_name` (str): Vectorizer choice ("tfidf" or "count")

**Returns:**
- `prediction` (str): Classification result ("FAKE" or "REAL")
- `confidence` (float): Confidence score (0.0 to 1.0)

#### `validate_input(text, title="")`

**Parameters:**
- `text` (str): Main text content
- `title` (str): Optional title

**Returns:**
- `is_valid` (bool): Whether input passes validation
- `result` (str): Cleaned text or error message

## 🎨 User Interface

The Streamlit interface provides:

- **Input Section**: Text area for news con