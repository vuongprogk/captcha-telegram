import logging
import os
import string

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, filters, ContextTypes, \
    ApplicationBuilder

# Characters used in CAPTCHA (including A-Z, 0-9, and special characters . and _)
characters = string.ascii_uppercase + string.digits + string.ascii_lowercase
model = load_model("model/captcha_model.keras")
img_path = 'images'

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def preprocess_captcha_image(image_path):
    """Preprocess the CAPTCHA image: grayscale, resize, normalize."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 80))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values to range [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (for grayscale)
    return np.expand_dims(img, axis=0)  # Add batch dimension for model input


def decode_predictions(predictions, num_characters=5):
    """Decode the model predictions into readable text."""
    predictions = predictions.reshape(num_characters, len(characters))  # Reshape to (num_characters, 36)
    predicted_chars = [characters[np.argmax(c)] for c in predictions]  # Choose max for each character
    return ''.join(predicted_chars)


# Define the /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! Send me a CAPTCHA image to solve.")


# Define the image handling function
async def solve_captcha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the image sent by the user and solve the CAPTCHA."""
    try:
        # Check if image folder exists, if not, create it
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        # Download the image
        file_path = img_path + "/image.png"
        await (await context.bot.getFile(update.message.photo[-1].file_id)).download_to_drive(file_path)

        # Preprocess the image for prediction
        preprocessed_image = preprocess_captcha_image(file_path)

        # Use the model to predict the CAPTCHA text
        predictions = model.predict(preprocessed_image)
        captcha_text = decode_predictions(predictions)

        # Send the solved CAPTCHA text back to the user
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Solved CAPTCHA: {captcha_text}")

    except Exception as e:
        logging.exception("Failed to process CAPTCHA: " + str(e))
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to solve CAPTCHA")


# Main function to set up the Telegram bot
def main() -> None:
    TOKEN = '7841472143:AAH-KxkS45vRxsaVLKtPKkfDjd4COv3NuU4'

    # Create the application and add handlers
    application = ApplicationBuilder().token(TOKEN).build()

    # Command handler for /start command
    start_handler = CommandHandler('start', start)

    # Message handler for images (CAPTCHA images)
    solver_handler = MessageHandler(filters.PHOTO, solve_captcha)

    # Add handlers to the application
    application.add_handler(start_handler)
    application.add_handler(solver_handler)

    # Start polling to listen for messages
    application.run_polling()


if __name__ == '__main__':
    main()
