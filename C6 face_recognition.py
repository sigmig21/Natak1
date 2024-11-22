import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# Actor to label mapping
y_vals = {
    "Brad Pitt": 0, "Hugh Jackman": 1, "Johnny Depp": 2, "Leonardo DiCaprio": 3,
    "Robert Downey Jr": 4, "Tom Cruise": 5, "Tom Hanks": 6, "Will Smith": 7,
}
DATASET_PATH = "C:\\Codes\\Assignments\\CVDL\\datasets\\CVDLDataset\\Face_recognition\\Celebrity Faces Dataset"

# Load dataset
def load_images_and_labels(path, labels):
    images, targets = [], []
    for actor, label in labels.items():
        actor_folder = os.path.join(path, actor)
        if os.path.exists(actor_folder):
            for img_name in os.listdir(actor_folder):
                img = cv2.imread(os.path.join(actor_folder, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (200, 200)))
                    targets.append(label)
    return np.array(images) / 255.0, np.array(targets)

X, y = load_images_and_labels(DATASET_PATH, y_vals)
X = np.expand_dims(X, -1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model definition
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 1)), MaxPool2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'), MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPool2D((2, 2)),
    Flatten(), Dense(8, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=y_vals.keys(), yticklabels=y_vals.keys())
plt.show()

# Save model
model.save("face_recognition_model.keras")

# Predict single image
def predict_actor(image_path, model, labels):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = cv2.resize(img, (200, 200)) / 255.0
    prediction = np.argmax(model.predict(np.expand_dims(img, (0, -1))))
    return {v: k for k, v in labels.items()}.get(prediction, "Unknown")

# Example usage
if __name__ == "__main__":
    test_image_path = "C:/Codes/Assignments/CVDL/datasets/CVDLDataset/Face_recognition/Celebrity Faces Dataset/Hugh Jackman/003_8889ec2c.jpg"  # Replace with your test image path
    print(f"The predicted actor is: {predict_actor(test_image_path, model, y_vals)}")
