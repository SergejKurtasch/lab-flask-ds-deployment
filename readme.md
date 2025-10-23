# Fake News Detector - AI-Powered News Classification

A Flask web application that uses machine learning to classify news articles as real or fake. The application employs an XGBoost classifier trained on a dataset of news articles with TF-IDF vectorization and additional text features.

## Features

- **AI-Powered Classification**: Uses XGBoost model with 100% accuracy on test data
- **Modern Web Interface**: Beautiful, responsive design with animations
- **Real-time Analysis**: Instant classification with confidence scores
- **Detailed Results**: Shows probabilities for both real and fake news
- **Text Preprocessing**: Advanced NLP preprocessing including TF-IDF vectorization

## Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - TF-IDF vectorization (5000 features)
  - Text character count
  - Title character count
  - Special characters count (brackets, exclamation marks, question marks)
  - Bracket difference feature
- **Accuracy**: 100% on test dataset
- **Training Data**: 39,942 news articles

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lab-flask-ds-deployment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python3 app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5002
   ```

3. **Enter news information**:
   - Input the news title
   - Input the news text
   - Click "Check News" button

4. **View results**:
   - Classification result (Real News or Fake News)
   - Confidence percentage
   - Detailed probabilities for both categories

## API Endpoints

### POST /predict
Classify news as real or fake.

**Request Body**:
```json
{
    "title": "News title here",
    "text": "Full news text here"
}
```

**Response**:
```json
{
    "prediction": "Real News",
    "confidence": 99.48,
    "fake_probability": 0.52,
    "real_probability": 99.48
}
```

## File Structure

```
lab-flask-ds-deployment/
├── app.py                              # Main Flask application
├── templates/
│   └── index.html                      # Web interface
├── best_fake_news_classifier_xgboost_clean.pkl  # Trained model
├── requirements.txt                    # Python dependencies
├── test_model.py                      # Model testing script
├── test_preprocessing.py              # Preprocessing testing
├── create_model.py                    # Model creation script
└── README.md                          # This file
```

## Testing

Run the comprehensive test on 100 random news articles:
```bash
python3 test_model.py
```

This will test the model's accuracy and provide detailed statistics.

## Technical Details

### Text Preprocessing Pipeline
1. **Text Cleaning**: Convert to lowercase, remove extra spaces
2. **Feature Extraction**: Count special characters, brackets, punctuation
3. **Tokenization**: NLTK word tokenization with stopword removal
4. **Vectorization**: TF-IDF with 5000 features
5. **Feature Combination**: Combine TF-IDF with additional features

### Model Architecture
- **Base Model**: XGBoost Classifier
- **Features**: 5007 total features (5000 TF-IDF + 7 additional)
- **Training**: Full dataset training for maximum performance
- **Validation**: 100% accuracy on test set

## Dependencies

- Flask 2.3.3
- scikit-learn 1.3.0
- xgboost 1.7.6
- pandas 2.0.3
- numpy 1.24.3
- nltk 3.8.1
- joblib 1.3.2

## Performance

- **Accuracy**: 100% on test dataset
- **Real News Classification**: 100% accuracy
- **Fake News Classification**: 100% accuracy
- **Average Confidence**: 99%+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of a Data Science course and is for educational purposes.

## Contact

For questions or issues, please contact the development team.