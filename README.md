# Next Word Prediction Using LSTM

## Project Overview

This project develops a deep learning model for predicting the next word in a given sequence using Long Short-Term Memory (LSTM) networks. The model is trained on Shakespeare's "Hamlet" text, making it capable of generating contextually appropriate next words based on input sequences.

## Features

- ✅ NLTK Gutenberg corpus integration for data collection
- ✅ Advanced text preprocessing with tokenization & padding
- ✅ Dual LSTM layer architecture with embedding & dropout
- ✅ Real-time next word prediction functionality
- ✅ Model persistence (HDF5 & Keras formats)
- ✅ Streamlit web application for deployment

## Project Workflow

### Step 1: Data Collection
```
- Download Shakespeare's "Hamlet" from NLTK Gutenberg corpus
- Save raw text to hamlet.txt file
- Dataset contains ~32K words of Shakespearean English
```

### Step 2: Data Preprocessing
```
- Tokenization: Convert text to lowercase → map unique words to indices (frequency-based)
- Sequence Generation: Create n-gram sequences from text (line by line)
- Max Length Calculation: Determine uniform sequence length for padding
- Padding: Apply pre-padding to all sequences using zeros
```

### Step 3: Data Preparation
```
- Split sequences into:
  • X (predictors): All words except the last in sequence
  • Y (labels): The last word in sequence (target to predict)
- One-Hot Encoding: Convert Y to categorical format with total_words classes
```

### Step 4: Train-Test Split
```
- 80% training data
- 20% testing data
- Stratified split using scikit-learn
```

### Step 5: Model Architecture

```
Input Layer
    ↓
Embedding Layer (100 dimensions)
    ↓
LSTM Layer (150 units, return_sequences=True)
    ↓
Dropout Layer (20%)
    ↓
LSTM Layer (100 units)
    ↓
Dense Output Layer (softmax, total_words units)
```

**Configuration:**
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Epochs: 50

### Step 6: Model Training
```
- Train on 80% of data (x_train, y_train)
- Validate on 20% of data (x_test, y_test)
- Monitor validation loss for overfitting detection
- Early stopping can be implemented for optimization
```

### Step 7: Next Word Prediction
```
Algorithm:
1. Tokenize input text sequence
2. Convert tokens to indices
3. Trim/pad to match model input length
4. Pass through model → get probability distribution
5. Extract word with highest probability (argmax)
6. Return predicted word
```

### Step 8: Model Persistence
```
- Save trained model in HDF5 format (.h5)
- Save trained model in Keras format (.keras)
- Pickle tokenizer for consistent preprocessing
```

### Step 9: Model Loading & Evaluation
```
- Load saved model from disk
- Evaluate on test dataset
- Use for predictions without retraining
```

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow/Keras
- NLTK
- Scikit-learn
- Numpy & Pandas
- Streamlit (optional, for web app)

### Setup
```bash
# Clone/navigate to project directory
cd "LSTM RNN Project"

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('gutenberg')"
```

## Usage

### Training the Model
```bash
# Run the Jupyter notebook
jupyter notebook lstm1.ipynb

# Or execute cell by cell in VS Code
```

### Using the Web Application
```bash
streamlit run app.py
```

### Making Predictions (Python)
```python
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model('lstm_hamlet_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict next word
input_text = "to be or not to be"
next_word = predict_next_word(model, tokenizer, input_text, max_length)
print(f"Next word: {next_word}")
```

## Project Files

```
LSTM RNN Project/
├── lstm1.ipynb                          # Main notebook with full pipeline
├── hamlet.txt                           # Shakespeare's Hamlet text
├── lstm_hamlet_model.h5                 # Trained model (HDF5 format)
├── lstm_hamlet_model_2.keras            # Trained model (Keras format)
├── tokenizer.pickle                     # Saved tokenizer
├── app.py                               # Streamlit web application
├── logs/                                # TensorBoard training logs
└── README.md                            # This file
```

## Model Performance

- **Architecture**: Embedding → LSTM(150) → Dropout(0.2) → LSTM(100) → Dense(softmax)
- **Training**: 50 epochs on 80% training data
- **Evaluation**: Accuracy & validation metrics tracked on 20% test data

## Key Insights

1. **Embedding Layer**: Reduces dimensionality from vocabulary size to 100 dense features
2. **Dual LSTM**: First layer captures long-term dependencies with sequence output; second layer performs final transformation
3. **Dropout**: Prevents overfitting by randomly deactivating 20% of neurons during training
4. **Tokenizer**: Essential for consistent text-to-index conversion during inference

## Example Usage

```
Input:  "to be or not"
Output: "to" / "be" / "or" (depending on context frequency)

Input:  "what a piece of work"
Output: "is" (most likely next word based on training)
```

## Future Enhancements

- [ ] Implement early stopping for training optimization
- [ ] Add beam search for multiple next-word predictions
- [ ] Support for custom text datasets
- [ ] Temperature scaling for prediction diversity
- [ ] Generate multi-word sequences instead of single next word
- [ ] Deploy to cloud platform (Streamlit Cloud, Heroku, etc.)

## Dependencies

See `requirements.txt` for full dependency list:
- tensorflow
- nltk
- scikit-learn
- numpy
- pandas
- streamlit (optional)

## License

Educational project for learning LSTM networks and sequence prediction.

## Author

Dinesh M
