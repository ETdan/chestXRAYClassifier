import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the model
model = load_model('chest_xray_classification_model.h5')

# Define class names
class_names = ['COVID-19', 'Tuberculosis', 'Pneumonia', 'Normal']

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array

# Function to make predictions
def predict_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    prediction_percentage = np.max(predictions) * 100

    print(f'Predicted Class: {predicted_class}')
    print(f'Prediction Percentage: {prediction_percentage:.2f}%')

    # Plot barplot of prediction probabilities
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, predictions[0] * 100)
    plt.xlabel('Class')
    plt.ylabel('Prediction Probability (%)')
    plt.title('Prediction Probabilities')
    plt.show()

# Example usage
if __name__ == '__main__':
    img_path = 'path_to_your_image.jpg'  # Replace with the path to your image
    predict_image(img_path)
