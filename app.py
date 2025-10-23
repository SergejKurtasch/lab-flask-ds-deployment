from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Load the saved model and components
def load_model():
    """Load the saved model and preprocessing components"""
    try:
        model_package = joblib.load('best_fake_news_classifier_xgboost_clean.pkl')
        return model_package
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model_package = load_model()

if model_package is None:
    print("Failed to load model. Please check if the model file exists.")
    exit(1)

# Extract components from the model package
model = model_package['model']
tfidf_vectorizer = model_package['tfidf_vectorizer']
additional_features = model_package['feature_names']

# Text processing functions (same as in the notebook)
def clean_text(text):
    """Clean text from extra characters and convert to lowercase"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra characters but preserve punctuation for counting
    text = text.strip()
    
    return text

def count_special_chars(text):
    """Count special characters in text"""
    if pd.isna(text) or text == "":
        return 0, 0, 0, 0
    
    text = str(text)
    
    # Count opening brackets
    open_brackets = text.count('(') + text.count('[') + text.count('{')
    
    # Count closing brackets
    closed_brackets = text.count(')') + text.count(']') + text.count('}')
    
    # Count exclamation marks
    exclamation_marks = text.count('!')
    
    # Count question marks
    question_marks = text.count('?')
    
    return open_brackets, closed_brackets, exclamation_marks, question_marks

def count_characters(text):
    """Count number of characters in text"""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text))

def preprocess_for_vectorization(text):
    """Prepare text for vectorization with stopword removal"""
    if pd.isna(text) or text == "":
        return ""
    
    # Tokenization
    tokens = word_tokenize(str(text))
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and short words
    filtered_tokens = [word for word in tokens 
                      if word.lower() not in stop_words 
                      and len(word) > 2 
                      and word.isalpha()]
    
    return ' '.join(filtered_tokens)

def preprocess_text(title, text):
    """Preprocess input text following the same algorithm as in the notebook"""
    try:
        # Clean text data
        title_clean = clean_text(title)
        text_clean = clean_text(text)
        
        # Create additional features
        special_chars = count_special_chars(text)
        open_brackets = special_chars[0]
        closed_brackets = special_chars[1]
        exclamation_marks = special_chars[2]
        question_marks = special_chars[3]
        
        # New feature: difference between opening and closing brackets
        bracket_difference = open_brackets - closed_brackets
        
        # Count characters
        text_char_count = count_characters(text)
        title_char_count = count_characters(title)
        
        # Prepare for vectorization
        title_processed = preprocess_for_vectorization(title_clean)
        text_processed = preprocess_for_vectorization(text_clean)
        
        # Combine title and text for vectorization
        combined_text = title_processed + ' ' + text_processed
        
        # Apply TF-IDF vectorization
        tfidf_matrix = tfidf_vectorizer.transform([combined_text])
        
        # Prepare additional features
        additional_features_array = np.array([[
            open_brackets,
            closed_brackets,
            bracket_difference,
            exclamation_marks,
            question_marks,
            text_char_count,
            title_char_count
        ]])
        
        # Combine TF-IDF features with additional features
        X_tfidf = tfidf_matrix.toarray()
        X_additional = additional_features_array
        
        # Combine all features
        X_combined = np.hstack([X_tfidf, X_additional])
        
        return X_combined
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    """Home page with the form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        title = data.get('title', '')
        text = data.get('text', '')
        
        # Validate input
        if not title or not text:
            return jsonify({'error': 'Title and text are required'}), 400
        
        # Preprocess the text
        X_processed = preprocess_text(title, text)
        
        if X_processed is None:
            return jsonify({'error': 'Error in text preprocessing'}), 500
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]
        
        # Convert prediction to readable format
        # Note: In the model, 0 = Fake News, 1 = Real News (based on test results)
        result = 'Real News' if prediction == 1 else 'Fake News'
        confidence = float(max(prediction_proba)) * 100
        
        # Return result
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'fake_probability': round(float(prediction_proba[0]) * 100, 2),
            'real_probability': round(float(prediction_proba[1]) * 100, 2)
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
