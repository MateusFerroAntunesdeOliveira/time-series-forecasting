import os

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Access the environment variables
INPUT_CSV_FILE_PATH = os.getenv("INPUT_CSV_FILE_PATH")
OUTPUT_CSV_FILE_PATH = os.getenv("OUTPUT_CSV_FILE_PATH")
OUTPUT_MERGED_CSV_FILE_PATH = os.getenv("OUTPUT_MERGED_CSV_FILE_PATH")
OUTPUT_MERGED_CSV_FILE_NAME = os.getenv("OUTPUT_MERGED_CSV_FILE_NAME")
