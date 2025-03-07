import os
import numpy as np
import cv2
import string
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Characters used in CAPTCHA (A-Z, 0-9)
characters = string.ascii_uppercase + string.digits + string.ascii_lowercase
num_classes = len(characters)

dataset_dir = "captcha_dataset"  # Path to your dataset directory


# Function to encode each character of CAPTCHA text into one-hot vectors
def encode_text(text):
    # Create a dictionary to map characters to integers
    char_to_int = {char: i for i, char in enumerate(characters)}

    # Convert each character in the text to its corresponding integer index
    encoded = [char_to_int[char] for char in text]

    # Convert the integer indices to one-hot encoding
    return to_categorical(encoded, num_classes=num_classes)


# Load and preprocess CAPTCHA dataset
def load_captcha_dataset(dataset_dir):
    images = []
    labels = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png"):
            # Extract label (text before the underscore)
            label = filename.split(".")[0]

            # Load and preprocess the image
            img = cv2.imread(os.path.join(dataset_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 80))  # Resize to match input size
            img = img.astype("float32") / 255.0  # Normalize pixel values
            images.append(img)

            # One-hot encode the label (sequence of characters)
            labels.append(encode_text(label))  # One-hot encode label for each character

    # Convert to numpy arrays
    X = np.array(images).reshape(-1, 80, 200, 1)  # Reshape to (height, width, channels)
    y = np.array(labels)

    # Ensure y has the correct shape: (num_samples, num_characters, num_classes)
    return X, y


# Load dataset
X, y = load_captcha_dataset(dataset_dir)

# Check shape of X and y to make sure everything is aligned
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam


def build_model(num_characters, num_classes):
    model = Sequential()
    model.add(Input(shape=(80, 200, 1)))  # Input layer explicitly defined
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer for each character (num_characters x num_classes)
    model.add(Dense(num_characters * num_classes, activation='softmax'))

    # Reshape output to (batch_size, num_characters, num_classes)
    model.add(Reshape((num_characters, num_classes)))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define model parameters
input_shape = (80, 200, 1)  # Input shape (height, width, channels)
num_characters = 5  # Number of characters per CAPTCHA (adjust based on your dataset)

model = build_model(num_characters, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Save the model
model.save('captcha_model.keras')
