# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import numpy as np


# Load the trained model
model = load_model("D:\\django_project\\main\\mlproject\\scripts\\brainModel.h5")

# Define the path to your single image
image_path = 'D:\\django_project\\main\\mlproject\\static\\images\\no_tumor\\2.jpg'  # Replace this with the path to your image

# Define class labels
class_labels = ['glioma', 'meningioma', 'no-tumor', 'pituitary']  # Replace with your actual class labels

# Load and preprocess the image
img = image.load_img(image_path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Rescale pixel values to [0, 1]

# Make prediction
predictions = model.predict(img_array)

# Get predicted class index
predicted_class_index = np.argmax(predictions[0])

# Get predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print("Predicted class:", predicted_class_label)
