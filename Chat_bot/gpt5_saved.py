import tensorflow as tf
import numpy as np

# Import the necessary components
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("word_predictor_model.h5")

# Example vocabulary and mappings (these should be the same as when the model was trained)
# Read the text data
file = open("input_1.txt", "r", encoding="utf-8")
text = file.read()
file.close()

# Step 1: Prepare the data
words = text.split()  # Split the text into words
vocab = sorted(set(words))  # Unique words in the vocabulary
word_to_int = {word: idx for idx, word in enumerate(vocab)}
int_to_word = {idx: word for idx, word in enumerate(vocab)}

# Function to predict the next word based on the given input sequence
def predict_next_word(model, input_sequence):
    # Convert the input sequence (string of words) to integers
    input_ints = [word_to_int[word] for word in input_sequence.split() if word in word_to_int]
    input_data = np.array(input_ints)  # Shape: (sequence_length,)
    
    # Reshape input for LSTM (samples, timesteps, features)
    input_data = input_data.reshape((1, len(input_data), 1))
    
    # Normalize the input (if needed, based on your original preprocessing)
    input_data = input_data / float(len(vocab))

    # Predict the next word (output)
    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)  # Get the index of the predicted word
    predicted_word = int_to_word[predicted_idx]
    return predicted_word

# Example prediction with user input
input_sequence = input("Enter Input: ")

# Get the predicted next word
predicted_word = predict_next_word(model, input_sequence)
print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")

# Generate a sequence of words based on the model
for i in range(20):  # Generate 100 words in sequence
    input_sequence += " " + predicted_word
    predicted_word = predict_next_word(model, input_sequence)
    print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")
