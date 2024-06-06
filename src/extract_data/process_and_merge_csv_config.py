import os

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Access the log configuration environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL")
LOG_LEVEL_FORMAT = os.getenv("LOG_LEVEL_FORMAT")

# Access the file environment variables
INPUT_PATH = os.getenv("INPUT_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
OUTPUT_MERGED_PATH = os.getenv("OUTPUT_MERGED_PATH")
OUTPUT_MERGED_FILENAME = os.getenv("OUTPUT_MERGED_FILENAME")
