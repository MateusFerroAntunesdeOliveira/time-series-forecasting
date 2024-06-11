import sys
import os
import pandas as pd

from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME
from utility.load_constants import target_column
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def check_missing_values(df):
    logger.debug("Checking for missing values")
    return df.isnull().sum()

# TODO:
# Create a function to impute values using some estrategy to fill the zero values
def impute_values(df):
    pass

# TODO:
# Create a function to plot outliers using Z-score
def plot_outliers(df, column_name):
    pass

def clean_data():
    logger.info("Cleaning data")

    merged_csv_file = utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_MERGED_FILENAME)
    df_all = utils.read_csv_file_as_dataframe(merged_csv_file)
    
    # Check for missing values
    missing_values = check_missing_values(df_all)
    logger.info(f"Missing values:\n{missing_values}")
    
    # Impute missing values
    impute_values(df_all)
    
    # Plot outliers
    plot_outliers(df_all, target_column)

    logger.info("Data cleaning completed")
