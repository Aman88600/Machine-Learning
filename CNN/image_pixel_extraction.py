import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

real_dir = 'real_images'  
ai_dir = 'AI_images'  

datagen = ImageDataGenerator(rescale=1.0/255.0)


root_dir = 'train_directory'


# Now, use `flow_from_directory` to load the images
train_data_gen = datagen.flow_from_directory(
    root_dir,  # Root directory that contains 'real_images' and 'AI_images'
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
    shuffle=True
)

# This will generate batches of images and their respective labels automatically
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# Define the CNN model with two output nodes for multi-class classification
model = Sequential([
    Input(shape=(200, 200, 3)),  # Input shape for RGB images
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='sigmoid')  # Output layer with 2 units (for real and AI images)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()
# Train the model on the images from the generator
history = model.fit(
    train_data_gen,
    steps_per_epoch=3,  # Number of batches per epoch, adjust as necessary
    epochs=1000  # Number of epochs
)

# Save the trained model
model.save("cnn_real_vs_ai.h5")
