import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import string

# Read the text data
file = open("input_1.txt", "r", encoding="utf-8")
text = file.read()
file.close()

# Step 1: Prepare the data
words = text.split()  # Split the text into words
vocab = sorted(set(words))  # Unique words in the vocabulary
word_to_int = {word: idx for idx, word in enumerate(vocab)}  # Mapping words to integers
int_to_word = {idx: word for idx, word in enumerate(vocab)}  # Mapping integers to words

# Step 2: Create input-output pairs for training
sequence_length = 5  # Length of the sequence to consider for prediction
X = []
y = []

# Generate sequences of 'sequence_length' words as input and the next word as output
for i in range(len(words) - sequence_length):
    X.append([word_to_int[words[i + j]] for j in range(sequence_length)])  # Input: sequence of words
    y.append(word_to_int[words[i + sequence_length]])  # Output: next word

X = np.array(X)
y = np.array(y)

# Step 3: Reshape input to be compatible with LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 4: Normalize the input data to be in the range [0, 1]
X = X / float(len(vocab))

# Step 5: Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
model.add(Dense(len(vocab), activation='softmax'))  # Output layer with softmax activation
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

# Step 6: Train the model
model.fit(X, y, epochs=70, batch_size=32)

# Step 7: Use the model to predict the next word
def predict_next_word(model, input_sequence):
    # Convert the input sequence (string of words) to integers
    input_ints = [word_to_int[word] for word in input_sequence.split()]
    input_data = np.array(input_ints)  # Shape: (sequence_length,)
    
    # Reshape input for LSTM (samples, timesteps, features)
    input_data = input_data.reshape((1, len(input_data), 1))
    
    # Normalize the input
    input_data = input_data / float(len(vocab))

    # Predict the next word (output)
    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)  # Get the index of the predicted word
    predicted_word = int_to_word[predicted_idx]
    return predicted_word

# Save the model
model.save("word_predictor_model.h5")

# Example prediction
input_sequence = input("Enter Input: ")

predicted_word = predict_next_word(model, input_sequence)
print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")

# Generate a sequence of words based on the model
for i in range(10):
    input_sequence += " " + predicted_word
    predicted_word = predict_next_word(model, input_sequence)
    print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")
