import os
import numpy as np
import cv2
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Characters used in CAPTCHA
characters = string.ascii_uppercase + string.digits
dataset_dir = "captcha_dataset"

# Function to encode CAPTCHA text into one-hot vectors
def encode_text(text):
    lb = LabelBinarizer().fit(list(characters))
    return lb.transform(list(text))


# Load and preprocess CAPTCHA dataset
def load_captcha_dataset(dataset_dir):
    images = []
    labels = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png"):
            # Extract label (text before the underscore)
            label = filename.split("_")[0]

            # Load and preprocess the image
            img = cv2.imread(os.path.join(dataset_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 80))  # Resize to match input size
            img = img.astype("float32") / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(encode_text(label))  # One-hot encode label

    # Convert to numpy arrays
    X = np.array(images).reshape(-1, 80, 200, 1)  # Reshape to (height, width, channels)
    y = np.array(labels)

    return X, y


# Load dataset
X, y = load_captcha_dataset(dataset_dir)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the model
def build_model(input_shape, num_characters, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_characters * num_classes, activation='softmax'))  # Output for each character
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define model
input_shape = (80, 200, 1)  # Input shape (height, width, channels)
num_characters = 5  # Number of characters per CAPTCHA
num_classes = 36  # Number of classes (A-Z, 0-9)

model = build_model(input_shape, num_characters, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('captcha_model.h5')
