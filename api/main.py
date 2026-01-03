# main.py

import logging
import os
from dotenv import load_dotenv
from machine_learning_models.config import settings
from machine_learning_models.models import train_model

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler()],
    )

    # Check if the output directory exists
    output_dir = settings.model_output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train the model
    train_model()