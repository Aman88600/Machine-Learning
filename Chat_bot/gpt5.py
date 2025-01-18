import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import string


file = open("input_1.txt", "r", encoding="utf-8")
text = file.read()
file.close()

words = text.split()  
vocab = sorted(set(words))  
word_to_int = {word: idx for idx, word in enumerate(vocab)}  
int_to_word = {idx: word for idx, word in enumerate(vocab)}  

sequence_length = 5  
X = []
y = []


for i in range(len(words) - sequence_length):
    X.append([word_to_int[words[i + j]] for j in range(sequence_length)]) 
    y.append(word_to_int[words[i + sequence_length]])  

X = np.array(X)
y = np.array(y)


X = X.reshape((X.shape[0], X.shape[1], 1))

X = X / float(len(vocab))

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
model.add(Dense(len(vocab), activation='softmax'))  # Output layer with softmax activation
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

model.fit(X, y, epochs=70, batch_size=32)


def predict_next_word(model, input_sequence):
 
    input_ints = [word_to_int[word] for word in input_sequence.split()]
    input_data = np.array(input_ints)  # Shape: (sequence_length,)
    
    
    input_data = input_data.reshape((1, len(input_data), 1))
    

    input_data = input_data / float(len(vocab))

    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)  
    predicted_word = int_to_word[predicted_idx]
    return predicted_word

model.save("word_predictor_model.h5")

input_sequence = input("Enter Input: ")

predicted_word = predict_next_word(model, input_sequence)
print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")

for i in range(10):
    input_sequence += " " + predicted_word
    predicted_word = predict_next_word(model, input_sequence)
    print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")
