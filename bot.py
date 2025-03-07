import logging
import os

import cv2
import numpy as np
from keras.saving.save import load_model
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, filters, ContextTypes, \
    ApplicationBuilder

from model.captcha_solving_model import characters

model = load_model("captcha_model.h5")
img_path = f'images'


# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
def preprocess_captcha_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 80))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (for grayscale)
    return np.expand_dims(img, axis=0)  # Add batch dimension for model input

def decode_predictions(predictions, num_characters=5):
    predictions = predictions.reshape(num_characters, 36)
    predicted_chars = [characters[np.argmax(c)] for c in predictions]
    return ''.join(predicted_chars)


# Define a function that will be used to handle incoming messages
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! Send me a CAPTCHA image to solve.")

async def solve_captcha(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Get the image from the message
    try:
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        await (await context.bot.getFile(update.message.photo[-1].file_id)).download_to_drive(img_path+ "/image.png")
    except Exception as e:
        logging.exception('Failed to download captcha' + str(e))
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to download captcha")
        return
    try:
        image_path = img_path + "/image.png"
        preprocessed_image = preprocess_captcha_image(image_path)

        # Use the model to predict the CAPTCHA text
        predictions = model.predict(preprocessed_image)
        captcha_text = decode_predictions(predictions)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Solved CAPTCHA: {captcha_text}")
    except Exception as e:
        logging.exception('Failed to solve captcha' + str(e))
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to solve captcha")
        return


def main() -> None:
    TOKEN = '7841472143:AAH-KxkS45vRxsaVLKtPKkfDjd4COv3NuU4'

    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    solver_handler = MessageHandler(filters.PHOTO, solve_captcha)

    application.add_handler(start_handler)
    application.add_handler(solver_handler)

    application.run_polling()



if __name__ == '__main__':
    main()
