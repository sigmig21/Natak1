# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# Load compressed MNIST dataset
data = np.load("C:\\Codes\\Assignments\\CVDL\\datasets\\CVDLDataset\\mnist_compressed.npz")

# Extract test and training data
X_test, y_test = data['test_images'], data['test_labels']
X_train, y_train = data['train_images'], data['train_labels']

# Function to display an image with its corresponding label
def show_img(image, label):
    plt.gray()  # Display in grayscale
    plt.title(f"Label: {label}")
    plt.imshow(image)
    plt.show()

# Load pre-trained CNN model
model = load_model('./cnn_model.keras')

# Extract names of convolutional and pooling layers for analysis
layer_names = [
    layer.name for layer in model.layers 
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D))
]

# Select a specific test image for visualization and prediction
x = X_test[999]  # Test image
y = y_test[999]  # Corresponding label
show_img(x, y)

# Reshape input to match model's expected input shape
x = x.reshape(1, 28, 56, 1)  # Batch size of 1, height 28, width 56, channels 1

# Predict the class of the test image
predictions = model.predict(x, verbose=0)
predicted_class = np.argmax(predictions[0])  # Get the predicted class
confidence = predictions[0][predicted_class]  # Confidence of the prediction
print(f'Predicted class: {predicted_class}, Confidence: {confidence}')

# Display model architecture
model.summary()

# Function to visualize the outputs of a specific layer
def visualize_layer_outputs(layer_index, input_data, grid_dims=(2, 4)):
    """
    Visualizes the outputs of a specified layer.
    :param layer_index: Index of the layer to visualize
    :param input_data: Input data to the model
    :param grid_dims: Tuple specifying grid dimensions for plotting
    """
    # Extract the output of the desired layer
    desired_output = model.layers[layer_index].output
    intermediate_model = Model(inputs=model.inputs, outputs=[desired_output])
    
    # Get the layer outputs for the input data
    layer_output = intermediate_model.predict(input_data)
    
    # Extract feature maps from the layer output
    feature_maps = layer_output[0]
    
    # Plot feature maps
    fig, axes = plt.subplots(*grid_dims, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[-1]:  # Ensure valid feature map index
            ax.imshow(feature_maps[:, :, i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Feature Map {i+1}")
    plt.tight_layout()
    plt.show()

# Visualize outputs of the first five layers
for i in range(4):
    print(f"Visualizing outputs for layer {i}...")
    visualize_layer_outputs(i, x)

# Visualize flattened output of dense layers
for i in range(4, 6):  # Layers 4 and 5 correspond to dense layers
    intermediate_model = Model(inputs=model.inputs, outputs=[model.layers[i].output])
    layer_output = intermediate_model.predict(x)
    flattened_output = layer_output.flatten()
    
    # Plot flattened output
    plt.plot(flattened_output)
    plt.title(f"Flattened Output of Layer {i}")
    plt.show()
    
    # Print index of the maximum value in the flattened output
    print(f"Max value at index: {flattened_output.argmax()}")
