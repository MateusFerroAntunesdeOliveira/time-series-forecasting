import os
import logging
import colorlog

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

OUTPUT_FILTERED_MERGED_FILENAME = os.getenv("OUTPUT_FILTERED_MERGED_FILENAME")

OUTPUT_SELECTED_FEATURES_FILENAME = os.getenv("OUTPUT_SELECTED_FEATURES_FILENAME")

# Create a color log formatter
formatter = colorlog.ColoredFormatter(
    LOG_LEVEL_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

# Configure the logger
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger()

# Remove all handlers from the root logger
root_logger = logging.getLogger()
if root_logger.handlers:
    root_logger.handlers.clear()

# Set the logging level for matplotlib and PIL to WARNING to avoid excessive debug output
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

# Create a new handler and set the colored formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
