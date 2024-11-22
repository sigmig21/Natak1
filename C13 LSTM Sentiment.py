import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
data = pd.read_csv('twitter_training.csv', header=None)
data.columns = ['Sr No', 'Ecommerce Website', 'Sentiment', 'Review']
data = data[['Sentiment', 'Review']]

# Reduce dataset size for testing
data = data.sample(frac=0.1, random_state=42)  # Use 10% of data

# Text preprocessing
data['Review'] = data['Review'].astype(str)  # Ensure all values are strings
data['Review'] = data['Review'].str.lower()
data['Review'] = data['Review'].str.replace(r'\brt\b', ' ', regex=True)
data['Review'] = data['Review'].apply(lambda x: ''.join([char if char.isalnum() or char.isspace() else '' for char in x]))

# Sentiment encoding
sentiment_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
data['Sentiment'] = data['Sentiment'].map(sentiment_map)

# Tokenization and padding
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Review'].values)
X = tokenizer.texts_to_sequences(data['Review'].values)
X = pad_sequences(X, maxlen=100)  # Limit sequence length to 100

# Convert Sentiment to categorical
Y = to_categorical(data['Sentiment'].values, num_classes=4)

# Build model
embed_dim = 128
model = Sequential()
model.add(Embedding(max_features, embed_dim))  # Removed input_length argument
model.add(SpatialDropout1D(0.4))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=5, batch_size=128, verbose=2)  # Reduce epochs and increase batch size

print("\nSentiment Prediction using LSTM")
maxlen = X.shape[1]
sentiment_classes = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

# Menu for testing
while True:
    print("\nMenu:")
    print("1. Enter a review")
    print("2. Exit")
    ch = input("Enter your choice (1/2): ")

    if ch == '1':
        ip = input("\nEnter a review: ")
        ip = ip.lower()
        ip = ''.join([char if char.isalnum() or char.isspace() else '' for char in ip])
        
        review_seq = tokenizer.texts_to_sequences([ip])
        review_seq = pad_sequences(review_seq, maxlen=maxlen, dtype='int32', value=0)
        
        sentiment = model.predict(review_seq, batch_size=1, verbose=2)[0]
        print(f"Predicted Sentiment: {sentiment_classes[np.argmax(sentiment)]}")
   
    elif ch == '2':
        break
    else:
        print("Invalid choice")
