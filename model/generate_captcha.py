import os
import random
import string

from captcha.image import ImageCaptcha


# Function to generate random CAPTCHA text
def random_captcha_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


# Function to generate CAPTCHA image and save it
def generate_captcha_image(captcha_text, image_dir="captcha_dataset"):
    image = ImageCaptcha(width=200, height=80)
    captcha_image = image.generate_image(captcha_text)
    captcha_image = captcha_image.convert("RGB")

    # Save the image
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_path = os.path.join(image_dir, f"{captcha_text}.png")
    captcha_image.save(image_path)

    return image_path, captcha_text


def generate_dataset(num_samples=1000):
    image_texts = []
    for _ in range(num_samples):
        captcha_text = random_captcha_text()
        generate_captcha_image(captcha_text)
        image_texts.append(captcha_text)

    print(f"Generated {num_samples} CAPTCHA images.")


if __name__ == "__main__":
    generate_dataset(1000)
