Next Word Prediction Using LSTM - Step-by-Step Breakdown

1. Data Collection
Download Shakespeare's "Hamlet" text using NLTK's Gutenberg corpus
Save the raw text to hamlet.txt file for local access
2. Data Preprocessing
Tokenization: Convert raw text to lowercase and create a tokenizer that maps each unique word to an index (based on frequency)
Sequence Generation: Split text into lines and create n-gram sequences (progressively longer sequences ending with each word)
Find Max Length: Determine the maximum sequence length for padding
Padding: Pad all sequences to uniform length using pre-padding (add zeros at the beginning)
3. Prepare Training Data
Split Features & Labels: Separate sequences into predictors (x) and target labels (y)
X: all words except the last one in each sequence
Y: the last word (target to predict)
One-Hot Encode Labels: Convert y into categorical format with num_classes = total_words
4. Train-Test Split
Divide preprocessed data into 80% training and 20% testing sets using scikit-learn
5. Build LSTM Model Architecture
Embedding Layer: Convert word indices to dense vectors (100 dimensions)
First LSTM Layer: 150 units with return_sequences=True to output full sequence
Dropout Layer: 20% dropout to prevent overfitting
Second LSTM Layer: 100 units (final LSTM layer)
Output Dense Layer: total_words units with softmax activation for probability distribution
6. Compile & Train
Compiler Settings: Use categorical crossentropy loss and Adam optimizer
Training: Train for 50 epochs with validation monitoring to detect overfitting
7. Make Predictions
Input a text sequence
Tokenize and pad it to match model input shape
Get model prediction (probability distribution)
Extract the word with highest probability using argmax
8. Save Model & Tokenizer
Save trained model in .h5 format (HDF5) and .keras format
Pickle the tokenizer for consistent text preprocessing during deployment
9. Model Loading & Evaluation
Load saved model to evaluate on test set
Can use for predictions without retraining
