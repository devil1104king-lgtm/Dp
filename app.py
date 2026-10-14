import telebot
import requests
import os
import time
import logging
from model_loader import generate_image
from PIL import Image
import io

# --- CONFIGURATION ---
API_TOKEN = os.getenv('BOT_TOKEN') # Render dashboard se token set karo
WEBHOOK_URL = None # For webhook mode, but polling is easier for free tier

# Initialize Bot
bot = telebot.TeleBot(API_TOKEN)

# Logging
logging.basicConfig(level=logging.INFO)

# --- HELPER FUNCTIONS ---

def create_mask_from_image(image_path):
    """
    Simple heuristic mask creation. 
    For production, use a segmentation model like SAM or U-Net trained on clothing.
    Here we assume the user sends an image and we mask the center/torso area roughly.
    Better approach: Use a pre-trained segmentation model.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Create a black mask (everything hidden)
    mask = Image.new("L", (width, height), 0)
    
    # Define a region to inpaint (e.g., torso area). 
    # Adjust these coordinates based on typical portrait orientation.
    # This is a Juggad: Masking the central vertical strip.
    left = width // 4
    top = height // 4
    right = width - (width // 4)
    bottom = height - (height // 4)
    
    # Draw white rectangle on mask (white means 'change this area')
    mask.paste(255, (left, top, right, bottom))
    
    mask_path = "mask.png"
    mask.save(mask_path)
    return mask_path

@bot.message_handler(func=lambda message: message.photo)
def handle_photo(message):
    try:
        chat_id = message.chat.id
        bot.send_chat_action(chat_id, 'upload_photo')
        bot.send_message(chat_id, "Processing image... Please wait. This may take 1-2 minutes.")
        
        # 1. Download Photo
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        with open("input.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
            
        # 2. Create Mask (Juggad Method)
        # Note: For better results, integrate a Segmentation Model like 'segment-anything' here.
        # But for now, we use a fixed box mask.
        mask_path = create_mask_from_image("input.jpg")
        
        # 3. Generate Image
        bot.send_chat_action(chat_id, 'typing')
        output_path = generate_image("input.jpg", mask_path)
        
        # 4. Send Result
        with open(output_path, "rb") as photo:
            bot.send_photo(chat_id, photo)
            
        # Cleanup
        os.remove("input.jpg")
        os.remove("mask.png")
        if os.path.exists("output.png"):
            os.remove("output.png")
            
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")
        logging.error(f"Error processing message: {e}")

@bot.message_handler(func=lambda message: True)
def all_messages(message):
    bot.reply_to(message, "Send me a photo to start!")

# --- RENDER KEEP-ALIVE SERVER ---
# Render free tier spins down after inactivity. This keeps it alive.
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "AI Undress Bot is Running!"

if __name__ == '__main__':
    # Run Flask server for keep-alive
    import threading
    def run_flask():
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    # Start Bot Polling
    print("Starting Bot...")
    bot.infinity_polling()
