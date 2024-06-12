import os
import pandas as pd

from utility.config import logger

def create_and_check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def join_file_path(path, filename):
    return os.path.join(path, filename)

def check_file_exists(filepath):
    return os.path.exists(filepath)

def read_csv_file_as_dataframe(file_path):
    logger.debug(f"Reading file {file_path}")
    return pd.read_csv(file_path, sep=";", encoding="utf-8")

def save_csv_file(df, filepath):
    logger.debug(f"Saving file {filepath}")
    df.to_csv(filepath, sep=";", index=False)
