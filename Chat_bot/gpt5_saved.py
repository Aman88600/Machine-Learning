import tensorflow as tf
import numpy as np


from tensorflow.keras.models import load_model

model = load_model("word_predictor_model.h5")


file = open("input_1.txt", "r", encoding="utf-8")
text = file.read()
file.close()


words = text.split()  
vocab = sorted(set(words))  
word_to_int = {word: idx for idx, word in enumerate(vocab)}
int_to_word = {idx: word for idx, word in enumerate(vocab)}


def predict_next_word(model, input_sequence):

    input_ints = [word_to_int[word] for word in input_sequence.split() if word in word_to_int]
    input_data = np.array(input_ints) 
    
    input_data = input_data.reshape((1, len(input_data), 1))
    
    input_data = input_data / float(len(vocab))
    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction) 
    predicted_word = int_to_word[predicted_idx]
    return predicted_word


input_sequence = input("Enter Input: ")

predicted_word = predict_next_word(model, input_sequence)
print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")


for i in range(20):  
    input_sequence += " " + predicted_word
    predicted_word = predict_next_word(model, input_sequence)
    print(f"Given the input sequence '{input_sequence}', the predicted next word is '{predicted_word}'")
