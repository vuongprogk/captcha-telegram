import os
import random
import string
import numpy as np
from captcha.image import ImageCaptcha
from sklearn.preprocessing import LabelEncoder
import cv2
import json


# Function to generate random CAPTCHA text
def random_captcha_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


# Function to generate CAPTCHA image and save it
def generate_captcha_image(captcha_text, image_dir="captcha_dataset"):
    image = image = ImageCaptcha(
    width=200,
    height=80,
    font_sizes=(40, 50),
)

    captcha_image = image.generate_image(captcha_text)
    captcha_image = captcha_image.convert("RGB")

    # Save the image
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_path = os.path.join(image_dir, f"{captcha_text}.png")
    captcha_image.save(image_path)

    return image_path, captcha_text


# Function to preprocess image for model input
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 80))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    return img


# Function to generate the dataset and save to file
def generate_dataset(num_samples=1000, image_dir="captcha_dataset", labels_file="captcha_labels.json"):
    image_paths = []
    labels = []
    label_encoder = LabelEncoder()

    # Collect unique labels to fit the label encoder
    unique_labels = set()
    for _ in range(num_samples):
        captcha_text = random_captcha_text()
        image_path, text = generate_captcha_image(captcha_text, image_dir)
        image_paths.append(image_path)
        labels.append(text)
        unique_labels.add(text)

    # Fit the label encoder to the unique labels
    label_encoder.fit(list(unique_labels))

    # Convert text labels to encoded labels
    encoded_labels = [label_encoder.transform([label])[0] for label in labels]

    # Preprocess the images and save them in a numpy array
    X = np.array(
        [preprocess_image(image_path) for image_path in image_paths])  # Shape: (num_samples, height, width, channels)
    X = X.reshape(-1, 80, 200, 1)  # Reshape to fit the model (height, width, channels)

    print(f"Generated {num_samples} CAPTCHA images and saved dataset.")


if __name__ == "__main__":
    generate_dataset(1000)  # You can change the number of samples here
