import sys
import os

import logging
import colorlog

import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from data_processing.data_processing_config import LOG_LEVEL, LOG_LEVEL_FORMAT, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME

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

# Create a new handler and set the colored formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def join_file_path(path, filename):
    return os.path.join(path, filename)

def read_csv_file_as_dataframe(file_path):
    logger.debug(f"Reading file {file_path}")
    return pd.read_csv(file_path)

def check_missing_values(df):
    logger.debug("Checking for missing values")
    return df.isnull().sum()

def apply_measures():
    logger.info("Applying measures")

    merged_csv_file = join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME)
    df_all = read_csv_file_as_dataframe(merged_csv_file)

    # Check for missing values
    missing_values = check_missing_values(df_all)
    logger.info(f"Missing values:\n{missing_values}")

    logger.info("Measures applied")
