import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

# Load the trained model
model = tf.keras.models.load_model("cnn_real_vs_ai.h5")

def on_drop(event):
    # Get the file path of the dropped image
    new_image_path = event.data
    try:
        # Load and preprocess the new image
        new_image = load_img(new_image_path, target_size=(200, 200))  # Resize image to 200x200
        new_image_array = img_to_array(new_image)  # Convert image to numpy array
        new_image_array = np.expand_dims(new_image_array, axis=0)  # Add batch dimension
        new_image_array = new_image_array / 255.0  # Normalize the image

        # Predict the class of the new image
        prediction = model.predict(new_image_array)

        # Display the result
        if prediction[0][0] > prediction[0][1]:
            prediction_label.config(text=f"Predicted: Real Image {prediction[0][0]}")
        else:
            prediction_label.config(text=f"Predicted: AI Image {prediction[0][1]}")

        # Display the image in Tkinter window
        img = Image.open(new_image_path)
        img = img.resize((400, 400))  # Resize image to fit within the window
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to the image
    except Exception as e:
        prediction_label.config(text="Invalid image file")

# Set up the Tkinter window
root = TkinterDnD.Tk()
root.title("Drag and Drop Image Prediction")
root.geometry("500x600")

# Create a label to display the image
image_label = tk.Label(root, text="Drag and Drop an Image Here", width=50, height=10, relief="solid")
image_label.grid(row=0, column=0)

# Create a label to display the prediction result
prediction_label = tk.Label(root, text="", width=50)
prediction_label.grid(row=2, column=0)

# Set up drag-and-drop functionality
image_label.drop_target_register(DND_FILES)
image_label.dnd_bind('<<Drop>>', on_drop)

# Start the Tkinter event loop
root.mainloop()
