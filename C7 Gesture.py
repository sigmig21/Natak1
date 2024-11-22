import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load datasets
train_images = np.load(r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\gesture_detection\train_gesture.npy")
train_labels = np.load(r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\gesture_detection\train_gesture_labels.npy")
val_images = np.load(r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\gesture_detection\validation_gesture.npy")
val_labels = np.load(r"C:\Codes\Assignments\CVDL\datasets\CVDLDataset\gesture_detection\validation_gesture_labels.npy")

# Normalize image data to [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Ensure images have the correct shape (batch_size, height, width, channels)
train_images = np.expand_dims(train_images, axis=-1)  # Add a channel dimension for grayscale
val_images = np.expand_dims(val_images, axis=-1)

# Get the number of unique classes
num_classes = len(np.unique(train_labels))

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(train_images.shape[1], train_images.shape[2], 1)),  # Include input shape
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    epochs=20,  # Adjust epochs based on dataset size
    batch_size=32,  # Adjust batch size based on hardware
    validation_data=(val_images, val_labels)
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot a sample image
import matplotlib.pyplot as plt

# Choose a sample image index
sample_idx = 4

# Get the true and predicted labels
true_label = np.argmax(val_labels[sample_idx])
predicted_label = np.argmax(model.predict(val_images[sample_idx].reshape(1, val_images.shape[1], val_images.shape[2], 1)))

# Plot the image and its true and predicted labels
plt.imshow(val_images[sample_idx].reshape(val_images.shape[1], val_images.shape[2]), cmap="gray")
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.show()
