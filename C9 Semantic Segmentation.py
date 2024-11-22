# Import necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define paths for image and label folders
image_folder = r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\semantic segmentation\Images"
label_folder = r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\semantic segmentation\Labels"

# Define the target size for resizing images
img_size = (128, 128)

# Initialize lists to hold images and their corresponding labels
images = []
labels = []

# Load and preprocess the input images
for img_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, img_name)  # Get full image path
    image = load_img(image_path, target_size=img_size)  # Load and resize image
    images.append(img_to_array(image) / 255.0)  # Normalize pixel values to [0, 1]

# Load and preprocess the labels
for label_name in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_name)  # Get full label path
    label = load_img(label_path, target_size=img_size, color_mode="grayscale")  # Load and resize label
    labels.append(img_to_array(label) / 255.0)  # Normalize pixel values to [0, 1]

# Expand dimensions of labels to match input shape
labels = np.expand_dims(labels, axis=-1)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Ensure training and label data have matching lengths
x_train, y_train = x_train[:min(len(x_train), len(y_train))], y_train[:min(len(x_train), len(y_train))]

# Build the U-Net model
# Input layer
inputs = Input(shape=(128, 128, 3))  # Expecting RGB input images

# Encoder
c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # Convolutional layer
p1 = MaxPooling2D((2, 2))(c1)  # Downsampling

c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)  # Convolutional layer
p2 = MaxPooling2D((2, 2))(c2)  # Downsampling

# Bottleneck
c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)  # Convolutional layer

# Decoder
u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c3)  # Upsampling
concat1 = Concatenate()([u1, c2])  # Skip connection
c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)  # Convolutional layer

u2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c4)  # Upsampling
concat2 = Concatenate()([u2, c1])  # Skip connection
c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)  # Convolutional layer

# Output layer
outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)  # Binary segmentation output

# Compile the model
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Train the model
history = model.fit(
    np.array(x_train), np.array(y_train),
    validation_data=(np.array(x_val), np.array(y_val)),
    epochs=10,
    batch_size=30
)

# Visualize a sample prediction (optional)
sample_img = x_val[5]
sample_label = y_val[5]
predicted_mask = model.predict(np.expand_dims(sample_img, axis=0))[0]

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(sample_img)

plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(sample_label.squeeze(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.show()
