from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the model
model = load_model('pneumonia_detection_model.h5')

# Define the Flask app
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for processing the uploaded image
@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image
    img = request.files['image']

    # Preprocess the image
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)

    # Get the class label
    class_label = np.argmax(predictions)

    # Get the confidence score
    confidence_score = predictions[0][class_label]

    # Use Grad-CAM to visualize the model's focus areas
    grad_cam = GradCAM(model, class_label)
    heatmap = grad_cam.compute_heatmap(img)
    heatmap = cv2.resize(heatmap, (224, 224))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.imshow(img[0], alpha=0.5)
    plt.savefig('static/heatmap.png')

    # Return the result
    return render_template('result.html', class_label=class_label, confidence_score=confidence_score)

# Define the route for displaying the result
@app.route('/result')
def result():
    return render_template('result.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
