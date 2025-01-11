from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Constants
IMAGE_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 3)

# Function to preprocess an input image
def preprocess_image(image_path):
    """
    Load and preprocess the image to 200x200 RGB.
    :param image_path: Path to the image file.
    :return: Preprocessed image array with shape (1, 200, 200, 3).
    """
    image = Image.open(image_path)
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 200x200
    image = image.resize(IMAGE_SIZE)
    
    # Convert to array and normalize
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# CNN Model
def create_model():
    """
    Build and compile a simple CNN model for binary classification.
    :return: Compiled CNN model.
    """
    model = Sequential([
        # Convolutional and Pooling Layers
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example Usage
if __name__ == "__main__":
    # Path to your image (update this with the actual image path)
    image_path = "image.png"

    # Preprocess the single image
    preprocessed_image = preprocess_image(image_path)

    # Create the CNN model
    model = create_model()

    # Normally, you'd train the model here or load a pre-trained model
    # For simplicity, we assume a saved model exists and load it
    model_path = "ai_image_classifier.h5"  # Path to saved model
    try:
        model.load_weights(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("No pre-trained model found. Please train the model first.")
        exit()

    # Predict
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] > 0.5:
        print("Prediction: AI-generated")
    else:
        print("Prediction: Not AI-generated")
